#!/usr/bin/env python
"""
evaluate_lora_final.py
----------------------

Evaluate a ToonCrafter checkpoint (optionally augmented with LoRA adapters) on
the ATD-12K test split **and** emit side-by-side comparison videos every *N*
samples.

Major features retained
~~~~~~~~~~~~~~~~~~~~~~~
* IDEFICS-style multi-image CLIP conditioning                (unchanged)
* DDIM sampling with text-and-image cross-attention          (unchanged)
* LPIPS (VGG), PSNR, SSIM metrics                            (unchanged)
* LoRA α∕r *scaling bug* patch after `PeftModel.from_pretrained` (unchanged)

No training / loss computation logic is present in this file, so your fine-
tuning objective is unaffected.
"""

from __future__ import annotations

# ──────────────────────────── std lib & third-party ────────────────────────────
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import cv2
import lpips
import numpy as np
import torch
import torchvision
from einops import rearrange
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from skimage.metrics import (peak_signal_noise_ratio as psnr,
                             structural_similarity as ssim)
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToPILImage
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel

# project-local imports
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custom_utils.datasets import ATD12K_Dataset            # noqa: E402
from lvdm.models.samplers.ddim import DDIMSampler           # noqa: E402
from utils.utils import instantiate_from_config             # noqa: E402

# ────────────────────────────────────────────────────────────────────────────────
#                                   Helpers
# ────────────────────────────────────────────────────────────────────────────────


def _load_base_and_lora(model: torch.nn.Module,
                        ckpt_path: str,
                        lora_scale: float,
                        lora_dir: str | None = None) -> torch.nn.Module:
    """
    Load a ToonCrafter checkpoint *and* (optionally) a LoRA adapter.

    The PEFT < 0.6 bug that drops α∕r during `from_pretrained` is patched
    in-place immediately after loading.
    """
    print(f"[INIT] Loading base model → {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("    ↪ missing:", ", ".join(k for k in missing if "lora" not in k))
    if unexpected:
        print("    ↪ unexpected:", ", ".join(unexpected))
    print("    ✔ base weights OK")

    model.is_lora = False
    
    if not lora_dir:
        return model

    cfg_file = Path(lora_dir) / "adapter_config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"{cfg_file} not found – is {lora_dir} correct?")

    l_cfg   = OmegaConf.load(cfg_file)
    print(l_cfg)
    l_alpha = l_cfg.get("lora_alpha", 16)
    l_rank  = l_cfg.get("rank", l_cfg.get("r", 16))

    print(f"[INIT] Attaching LoRA ({l_alpha}/{l_rank}) → {lora_dir}")
    model.model.diffusion_model = PeftModel.from_pretrained(
        model.model.diffusion_model, lora_dir, is_trainable=False
    )

    # Patch α∕r scaling
    for mod in model.model.diffusion_model.modules():
        if hasattr(mod, "lora_A") and hasattr(mod, "r"):
            # mod.scaling = l_alpha / mod.r
            # rank = mod.r['default'] if isinstance(mod.r, dict) else mod.r
            # mod.scaling = l_alpha / rank
            # --- Start of Replacement ---
            rank = mod.r['default'] if isinstance(mod.r, dict) else mod.r
            scaling_value = l_alpha / rank
            mod.scaling = {'default': scaling_value}
            # print("Before LoRA Scaling:", mod.scaling['default'])
            mod.scaling['default'] *= lora_scale
            # print("After LoRA Scaling:", mod.scaling['default'])
            # --- End of Replacement ---
    print("    ✔ LoRA attached\n")
    model.is_lora = True
    return model


@torch.no_grad()
def _encode_to_latent(model: torch.nn.Module, vid: torch.Tensor) -> torch.Tensor:
    """
    Encode B×3×T×H×W RGB video (−1…1) to latent space.
    """
    b, _, t, *_ = vid.shape
    z = model.encode_first_stage(rearrange(vid, "b c t h w -> (b t) c h w"))
    return rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)


@torch.no_grad()
def _clip_emb(batch_pil: List, clip_m: CLIPModel,
              clip_p: CLIPProcessor, dev: torch.device) -> torch.Tensor:
    """
    Obtain CLIP visual embeddings for a list of PIL images.
    """
    tokens = clip_p(images=batch_pil, return_tensors="pt").to(dev)
    return clip_m.vision_model(**tokens).last_hidden_state


def _tensor_to_pil(x: torch.Tensor) -> List:
    """
    Convert B×3×H×W tensor (−1…1) → List[PIL.Image].
    """
    imgs = ((x.cpu() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
    return [ToPILImage()(img) for img in imgs]


def _sample_ddim(model, clip_p, clip_m, prompt, pair, noise_shape,
                 steps, scale):
    """
    Run DDIM sampling conditioned on (text, start, end) frames.
    """
    sampler = DDIMSampler(model)
    dev     = model.device
    b, _, _, h, w = pair.shape

    # CLIP context (batch-safe)
    img_ctx = model.image_proj_model(
        _clip_emb(_tensor_to_pil(pair[:, :, 0]), clip_m, clip_p, dev)
    )
    zero_pil = ToPILImage()(torch.zeros(3, h, w, dtype=torch.uint8))
    img_uc   = model.image_proj_model(
        _clip_emb([zero_pil] * b, clip_m, clip_p, dev)
    )

    txt_ctx = model.get_learned_conditioning(prompt)
    txt_uc  = model.get_learned_conditioning([""] * b)

    # Latent concat (keep t = 0, 1)
    z      = _encode_to_latent(model, pair)
    c_cat  = torch.zeros(noise_shape, device=dev)
    c_cat[:, :, 0], c_cat[:, :, -1] = z[:, :, 0], z[:, :, 1]

    cond = {"c_crossattn": [torch.cat([txt_ctx, img_ctx], 1)],
            "c_concat":    [c_cat]}
    uc   = {"c_crossattn": [torch.cat([txt_uc,  img_uc], 1)],
            "c_concat":    [c_cat]}

    vid_z, _ = sampler.sample(
        S=steps, conditioning=cond, batch_size=b, shape=noise_shape[1:],
        verbose=False, unconditional_guidance_scale=scale,
        unconditional_conditioning=uc, eta=0.0,
    )
    return model.decode_first_stage(vid_z)


# ────────────────────────────────────────────────────────────────────────────────
#                                    Main
# ────────────────────────────────────────────────────────────────────────────────


def main(cfg: argparse.Namespace) -> None:
    """Entry-point: load nets → iterate loader → compute metrics (+optional mp4)."""
    seed_everything(cfg.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load networks ──────────────────────────────────────────────────────────
    conf = OmegaConf.load(cfg.config)
    net  = instantiate_from_config(conf.get("model", OmegaConf.create())).to(dev)
    net  = _load_base_and_lora(net, cfg.ckpt_path, cfg.lora_scale, cfg.lora_ckpt_dir)
    net.eval()

    clip_m  = CLIPModel.from_pretrained(cfg.clip_model_name).to(dev)
    clip_p  = CLIPProcessor.from_pretrained(cfg.clip_model_name)
    lpips_f = lpips.LPIPS(net="vgg").to(dev).eval()

    # ── data ───────────────────────────────────────────────────────────────────
    ds = ATD12K_Dataset(cfg.dataset_path, video_size=(cfg.height, cfg.width),
                        split="test")
    if cfg.debug:
        ds = Subset(ds, range(cfg.bs * 2))

    dl = DataLoader(ds, batch_size=cfg.bs, shuffle=False,
                    num_workers=0 if cfg.debug else 4, pin_memory=True)

    out_dir = Path(cfg.output_dir)
    vid_dir = out_dir / "evaluation_videos"
    if cfg.save_every_n > 0:
        vid_dir.mkdir(parents=True, exist_ok=True)

    # ── metrics ────────────────────────────────────────────────────────────────
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dl, desc="Evaluating", ncols=90)):
            b = batch["start_frame"].size(0)

            ds_fact = getattr(net.first_stage_model.encoder, "downsample_factor",
                              getattr(net.first_stage_model.encoder,
                                      "current_downsample_factor", 8))
            lat_h, lat_w = cfg.height // ds_fact, cfg.width // ds_fact
            noise_shape  = [b, net.model.diffusion_model.out_channels,
                            cfg.video_length, lat_h, lat_w]

            start = batch["start_frame"].to(dev)
            end   = batch["end_frame"].to(dev)
            gt_mid = batch["ground_truth_middle"].to(dev)

            gen_vid = _sample_ddim(net, clip_p, clip_m, batch["prompt"],
                                   torch.stack([start, end], 2),
                                   noise_shape, cfg.ddim_steps,
                                   cfg.guidance_scale)
            gen_mid = gen_vid[:, :, cfg.video_length // 2]

            # ─ metrics loop (per-sample) ───────────────────────────────────────
            for j in range(b):
                gt_rgb  = ((gt_mid[j].permute(1, 2, 0).cpu() * 0.5 + 0.5)
                           * 255).byte().numpy()
                gen_rgb = ((gen_mid[j].permute(1, 2, 0).cpu() * 0.5 + 0.5)
                           * 255).byte().numpy()

                psnr_vals.append(psnr(gt_rgb, gen_rgb, data_range=255))

                ssim_raw = ssim(cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2GRAY),
                                cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY),
                                data_range=255)
                ssim_vals.append(ssim_raw if np.isfinite(ssim_raw) else 0.0)

            lpips_batch = lpips_f(gt_mid.clamp(-1, 1), gen_mid.clamp(-1, 1))
            lpips_vals.extend(lpips_batch.flatten().cpu().numpy())

            # ─ optional mp4 ────────────────────────────────────────────────────
            if cfg.save_every_n > 0 and idx % cfg.save_every_n == 0:
                gt_stack  = torch.stack([start[0], gt_mid[0], end[0]], 1)
                # gen_stack = torch.cat([start[0:1],
                #                        gen_vid[0, :, 1:-1],
                #                        end[0:1]], 1)
                gen_stack = torch.stack([start[0], gen_mid[0], end[0]], 1)
                grid = torch.cat([gt_stack, gen_stack], -2)        # vertical
                grid_u8 = (((grid.cpu() + 1) / 2 * 255)
                           .clamp(0, 255).byte().permute(1, 2, 3, 0))
                torchvision.io.write_video(
                    vid_dir / f"sample_{idx * cfg.bs}.mp4",
                    grid_u8, fps=3, video_codec="h264"
                )

    tag = Path(cfg.lora_ckpt_dir).name if cfg.lora_ckpt_dir else "Baseline"
    print(f"\nEVALUATION COMPLETE • {tag}")
    print(f"  PSNR : {np.mean(psnr_vals):6.2f}")
    print(f"  SSIM : {np.mean(ssim_vals):7.4f}")
    print(f"  LPIPS: {np.mean(lpips_vals):7.4f}\n")


# ────────────────────────────────────────────────────────────────────────────────
#                                CLI wrapper
# ────────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate ToonCrafter (+LoRA) on ATD-12K."
    )
    g = p.add_argument_group("Paths")
    g.add_argument("--output_dir", required=True)
    g.add_argument("--dataset_path", required=True)
    g.add_argument("--config", required=True)
    g.add_argument("--ckpt_path", required=True)
    g.add_argument("--lora_ckpt_dir")
    g.add_argument("--clip_model_name",
                   default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    g = p.add_argument_group("Eval opts")
    g.add_argument("--bs", type=int, default=1)
    g.add_argument("--debug", action="store_true")
    g.add_argument("--seed", type=int, default=123)
    g.add_argument("--save_every_n", type=int, default=0)

    g = p.add_argument_group("Model & sampling")
    g.add_argument("--height", type=int, default=320)
    g.add_argument("--width",  type=int, default=512)
    g.add_argument("--video_length", type=int, default=16)
    g.add_argument("--ddim_steps", type=int, default=50)
    g.add_argument("--guidance_scale", type=float, default=7.5)

    g.add_argument("--lora_scale", type=float, default=1.0, help="Scaling factor for LoRA adapters at inference time.")
    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())