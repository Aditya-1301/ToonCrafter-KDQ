import os
import csv
import time
import argparse
import random
import shutil
import multiprocessing

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from pytorch_lightning import seed_everything
from einops import rearrange

from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms.functional import to_pil_image

# avoid fork issues with DataLoader
multiprocessing.set_start_method("spawn", force=True)

from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset


def load_base_model(cfg_path, ckpt_path, device):
    cfg   = OmegaConf.load(cfg_path)
    model = instantiate_from_config(cfg.model).to(device)
    sd    = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = {}
    for k,v in sd.items():
        nk = k.replace("model.model.diffusion_model.", "model.diffusion_model.")
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=False)
    print(">>> Loaded base model")
    return model

def get_clip_embedder(device):
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").vision_model.to(device)
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return clip, proc, clip.config.hidden_size

def encode_clip(clip, proc, imgs, device, adapter=None):
    pil = [to_pil_image(img) for img in imgs.cpu()]
    inputs = proc(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        feats = clip(**inputs).last_hidden_state[:,0]  # [B, hidden_dim]
    if adapter:
        feats = adapter(feats)
    return feats

def get_unet(model):
    return model.model.diffusion_model

def get_latent_z(model, videos):
    b,c,t,h,w = videos.shape
    x = rearrange(videos, "b c t h w -> (b t) c h w")
    with torch.no_grad():
        z = model.encode_first_stage(x)
    return rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)

def compute_metrics(img1, img2):
    np1 = img1.cpu().numpy().transpose(0,2,3,1)
    np2 = img2.cpu().numpy().transpose(0,2,3,1)
    ps, ss = [], []
    for a,b in zip(np1,np2):
        ps.append(peak_signal_noise_ratio(a,b,data_range=1.0))
        ss.append(structural_similarity(a,b,data_range=1.0,channel_axis=2))
    return float(np.nanmean(ps)), float(np.nanmean(ss))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_path",  required=True)
    p.add_argument("--ckpt_path",     required=True)
    p.add_argument("--config",        required=True)
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--bs",            type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--lora_rank",     type=int,   default=16)
    p.add_argument("--alpha",         type=float, default=1.0)
    p.add_argument("--beta",          type=float, default=1.0)
    p.add_argument("--val_frac",      type=float, default=0.1)
    p.add_argument("--height",        type=int,   default=512)
    p.add_argument("--width",         type=int,   default=512)
    p.add_argument("--log_img_every", type=int,   default=1)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    seed_everything(args.seed, workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    writer   = SummaryWriter(os.path.join(args.output_dir, "tb"))
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_mse","train_lpips","train_psnr","train_ssim",
            "val_mse","val_lpips","val_psnr","val_ssim","lr"
        ])

    model      = load_base_model(args.config, args.ckpt_path, device)
    clip,proc,clip_dim = get_clip_embedder(device)
    ipm = model.image_proj_model
    exp = ipm.proj_in.in_features if hasattr(ipm, "proj_in") else None
    adapter = torch.nn.Linear(clip_dim, exp).to(device) if exp and exp != clip_dim else None

    unet = get_unet(model)
    for n,p in unet.named_parameters():
        p.requires_grad = "temporal_transformer" not in n
    for p in model.image_proj_model.parameters():
        p.requires_grad = True

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["to_q","to_k","to_v","to_out.0"],
        lora_dropout=0.1,
        bias="none"
    )
    lora_unet = get_peft_model(unet, lora_cfg).to(device)
    lora_unet.print_trainable_parameters()

    full = ATD12K_Dataset(args.dataset_path,
                         video_size=(args.height,args.width),
                         split="train")
    n_val = int(len(full)*args.val_frac)
    train_ds, val_ds = random_split(
        full, [len(full)-n_val, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    dl_kwargs = dict(batch_size=args.bs, num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    print(f"Train/Val: {len(train_ds)}/{len(val_ds)}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, lora_unet.parameters()),
        lr=args.lr
    )
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    best_lp  = float("inf")

    for epoch in range(1, args.epochs+1):
        # TRAIN
        lora_unet.train()
        t_mse=t_lp=t_ps=t_ss=0
        for batch in tqdm(train_loader, desc=f"[Train] E{epoch}"):
            optimizer.zero_grad()
            s = batch["start_frame"].to(device)
            m = batch["ground_truth_middle"].to(device)
            e = batch["end_frame"].to(device)

            # correct empty-middle objective
            z_se = get_latent_z(model, torch.stack([s,e],dim=2))
            z_m  = get_latent_z(model, m.unsqueeze(2))
            x = torch.zeros((s.size(0), z_se.size(1), 3, *z_se.shape[-2:]), device=device)
            x[:,:,0], x[:,:,2] = z_se[:,:,0], z_se[:,:,1]

            # conditioning
            txt = model.get_learned_conditioning(batch["prompt"])
            feat = encode_clip(clip, proc, s, device, adapter)  # [B,hidden]
            feat = feat.unsqueeze(1)                            # [B,1,hidden]
            ctx_img = model.image_proj_model(feat)              # [B,seq,latent]
            ctx = torch.cat([txt, ctx_img], dim=1)

            # forward + reconstruct
            t = torch.randint(0, model.num_timesteps, (s.size(0),), device=device)
            noise = torch.randn_like(x)
            noisy = model.q_sample(x_start=x, t=t, noise=noise)
            inp = torch.cat([noisy, x], dim=1)
            pred_noise = lora_unet(inp, t, context=ctx)
            rec = model.predict_start_from_noise(noisy, t, pred_noise)
            z_mid_pred = rec[:,:,1]
            img_pred = model.decode_first_stage(z_mid_pred).add(1).div(2).clamp(0,1)
            img_gt   = model.decode_first_stage(z_m[:,:,0]).add(1).div(2).clamp(0,1)

            mse = F.mse_loss(img_pred, img_gt)
            lp  = lpips_fn(img_pred, img_gt).mean()
            loss = args.alpha*mse + args.beta*lp
            loss.backward(); torch.nn.utils.clip_grad_norm_(lora_unet.parameters(),1.0)
            optimizer.step()

            ps, ss = compute_metrics(img_gt, img_pred)
            t_mse+=mse.item(); t_lp+=lp.item(); t_ps+=ps; t_ss+=ss

        n = len(train_loader)
        writer.add_scalar("Train/MSE",   t_mse/n, epoch)
        writer.add_scalar("Train/LPIPS", t_lp/n, epoch)
        writer.add_scalar("Train/PSNR",  t_ps/n, epoch)
        writer.add_scalar("Train/SSIM",  t_ss/n, epoch)
        writer.add_scalar("Train/LR",    optimizer.param_groups[0]["lr"], epoch)

        # VALIDATION
        lora_unet.eval()
        v_mse=v_lp=v_ps=v_ss=0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"[Val]   E{epoch}"):
                s = batch["start_frame"].to(device)
                m = batch["ground_truth_middle"].to(device)

                z_se = get_latent_z(model, torch.stack([s,batch["end_frame"].to(device)],dim=2))
                z_m  = get_latent_z(model, m.unsqueeze(2))
                x = torch.zeros_like(z_se).repeat(1,1,3,1,1)
                x[:,:,0], x[:,:,2] = z_se[:,:,0], z_se[:,:,1]

                txt = model.get_learned_conditioning(batch["prompt"])
                feat = encode_clip(clip, proc, s, device, adapter).unsqueeze(1)
                ctx_img = model.image_proj_model(feat)
                ctx = torch.cat([txt, ctx_img], dim=1)

                t = torch.randint(0, model.num_timesteps, (s.size(0),), device=device)
                noisy = model.q_sample(x_start=x, t=t, noise=torch.randn_like(x))
                inp = torch.cat([noisy,x], dim=1)
                pred_noise = lora_unet(inp, t, context=ctx)
                rec = model.predict_start_from_noise(noisy, t, pred_noise)
                z_mid_pred = rec[:,:,1]
                img_pred = model.decode_first_stage(z_mid_pred).add(1).div(2).clamp(0,1)
                img_gt   = model.decode_first_stage(z_m[:,:,0]).add(1).div(2).clamp(0,1)

                mse = F.mse_loss(img_pred, img_gt)
                lp  = lpips_fn(img_pred, img_gt).mean()
                ps, ss = compute_metrics(img_gt, img_pred)
                v_mse+=mse.item(); v_lp+=lp.item(); v_ps+=ps; v_ss+=ss

        vn = len(val_loader)
        writer.add_scalar("Val/MSE",   v_mse/vn, epoch)
        writer.add_scalar("Val/LPIPS", v_lp/vn, epoch)
        writer.add_scalar("Val/PSNR",  v_ps/vn, epoch)
        writer.add_scalar("Val/SSIM",  v_ss/vn, epoch)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                t_mse/n, t_lp/n, t_ps/n, t_ss/n,
                v_mse/vn, v_lp/vn, v_ps/vn, v_ss/vn,
                optimizer.param_groups[0]["lr"]
            ])

        if epoch % args.log_img_every == 0:
            grid = torch.cat([s, img_pred, img_gt, batch["end_frame"].to(device)], dim=-1)
            writer.add_image("Grid", grid[0], epoch, dataformats="CHW")
            od = os.path.join(args.output_dir, "images"); os.makedirs(od, exist_ok=True)
            import torchvision
            torchvision.utils.save_image(grid, f"{od}/epoch{epoch:03d}.png", nrow=1)

        for tag in ("", "_best"):
            d = os.path.join(args.output_dir, "lora"+tag)
            if os.path.isdir(d): shutil.rmtree(d)
            lora_unet.save_pretrained(d)

        if (v_lp/vn) < best_lp:
            best_lp = v_lp/vn

    writer.close()
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
