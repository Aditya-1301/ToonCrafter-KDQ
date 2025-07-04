import argparse
import os
import sys
import torch
import torchvision
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from peft import PeftModel
from einops import rearrange
from torchvision.transforms import ToPILImage
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import kornia

# Project root
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..', '..'))
from custom_utils.datasets import ATD12K_Dataset
from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config

def load_model_checkpoint(model, ckpt_path, lora_dir=None):
    sd = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)
    print(f">>> Loaded checkpoint from {ckpt_path} (non-strict)")
    if lora_dir and os.path.exists(lora_dir):
        print(f">>> Applying LoRA from {lora_dir}")
        base_unet = model.model.diffusion_model
        model.model.diffusion_model = PeftModel.from_pretrained(base_unet, lora_dir)
    return model

def get_latent_z(model, videos):
    B, C, T, H, W = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    return rearrange(z, '(b t) c h w -> b c t h w', b=B, t=T)

def preprocess_image(tensor):
    img = tensor.detach().cpu()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    img = (img * 0.5 + 0.5).clamp(0,1)
    return ToPILImage()(img)

def get_clip_tokens(pil_img, clip_model, clip_proc, device):
    inp = clip_proc(images=pil_img, return_tensors='pt', size=224).to(device)
    with torch.no_grad():
        out = clip_model.vision_model(**inp)
    return out.last_hidden_state  # [1, seq_len, 1280]

def correct_image_guided_synthesis(
    model, clip_proc, clip_model,
    prompts, videos, noise_shape,
    ddim_steps, guidance_scale, fs
):
    sampler = DDIMSampler(model)
    device = model.device
    B = noise_shape[0]
    latent_C, T, Hf, Wf = noise_shape[1:]

    # Image tokens + projection
    img = videos[:, :, 0]  # [B,C,H,W]
    tok = get_clip_tokens(preprocess_image(img), clip_model, clip_proc, device)
    img_emb = model.image_proj_model(tok)  # [B,seq,latent_C]

    blank = torch.zeros_like(img[0])
    tok0 = get_clip_tokens(preprocess_image(blank), clip_model, clip_proc, device)
    uc_img_emb = model.image_proj_model(tok0)

    # Text embeddings
    txt_emb = model.get_learned_conditioning(prompts)
    uc_txt  = model.get_learned_conditioning(['']*B)

    # Latent concat padding
    z = get_latent_z(model, videos)  # [B,latent_C,2,Hf,Wf]
    pad = torch.zeros((B, latent_C, T, Hf, Wf), device=device)
    pad[:, :, 0] = z[:, :, 0]
    pad[:, :, T-1] = z[:, :, 1]

    # Build conditioning
    cond = {
        "c_crossattn": [ torch.cat([txt_emb, img_emb], dim=1) ],
        "c_concat":    [ pad ]
    }
    uc = {
        "c_crossattn": [ torch.cat([uc_txt, uc_img_emb], dim=1) ],
        "c_concat":    [ pad ]
    }

    # DDIM sampling
    samples, _ = sampler.sample(
        S=ddim_steps,
        conditioning=cond,
        unconditional_conditioning=uc,
        unconditional_guidance_scale=guidance_scale,
        batch_size=B,
        shape=noise_shape[1:],
        fs=torch.tensor([fs]*B, device=device),
        verbose=False
    )

    return model.decode_first_stage(samples)  # [B,C,T,Hf,Wf]

def main(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load ToonCrafter
    cfg = OmegaConf.load(args.config)
    mc  = cfg.pop("model", OmegaConf.create())
    model = instantiate_from_config(mc).to(device)
    model = load_model_checkpoint(model, args.ckpt_path, args.lora_ckpt_dir)
    model.eval()

    # CLIP-H/14
    clip_model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    clip_proc  = CLIPProcessor.from_pretrained(args.clip_model_name)

    # Data loader
    ds     = ATD12K_Dataset(args.dataset_path, video_size=(args.height,args.width), split='test')
    sub    = torch.utils.data.Subset(ds, list(range(args.num_samples)))
    loader = DataLoader(sub, batch_size=1, shuffle=False, num_workers=2)

    sampler    = DDIMSampler(model)
    tag        = os.path.basename(args.lora_ckpt_dir) if args.lora_ckpt_dir else "baseline"
    save_dir   = os.path.join(args.output_dir, tag); os.makedirs(save_dir, exist_ok=True)
    noise_shape= [
        1,
        model.model.diffusion_model.out_channels,
        args.video_length,
        args.height // args.scale_factor,
        args.width  // args.scale_factor
    ]

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Visualizing")):
            start = batch['start_frame'].to(device)
            end   = batch['end_frame'].to(device)
            vids  = torch.stack([start, end], dim=2)  # [1,C,2,H,W]
            prom  = [batch['prompt'][0]]

            gen_vid = correct_image_guided_synthesis(
                model, clip_proc, clip_model,
                prom, vids, noise_shape,
                args.ddim_steps,
                args.unconditional_guidance_scale,
                args.fs
            )

            gt = torch.stack([
                batch['start_frame'][0],
                batch['ground_truth_middle'][0],
                batch['end_frame'][0]
            ], dim=1)  # [C,3,H,W]

            # Clamp + cast to uint8 to avoid float16 clamp errors and overflow flares
            gen_cpu = (gen_vid[0].cpu().float().permute(1,2,3,0).add(1).div(2).mul(255)
                       .clamp(0,255).to(torch.uint8))
            gt_cpu  = (gt.cpu().float().permute(1,2,3,0).add(1).div(2).mul(255)
                       .clamp(0,255).to(torch.uint8))

            torchvision.io.write_video(
                os.path.join(save_dir, f"sample_{i}_gen.mp4"),
                gen_cpu, fps=8, video_codec='h264'
            )
            torchvision.io.write_video(
                os.path.join(save_dir, f"sample_{i}_gt.mp4"),
                gt_cpu, fps=3, video_codec='h264'
            )

    print("✅ Visualization complete.")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",               type=str, required=True)
    p.add_argument("--lora_ckpt_dir",            type=str, default="")
    p.add_argument("--num_samples",              type=int, default=5)
    p.add_argument("--dataset_path",             type=str, required=True)
    p.add_argument("--config",                   type=str, required=True)
    p.add_argument("--ckpt_path",                type=str, required=True)
    p.add_argument("--height",                   type=int, default=320)
    p.add_argument("--width",                    type=int, default=512)
    p.add_argument("--video_length",             type=int, default=16)
    p.add_argument("--ddim_steps",               type=int, default=50)
    p.add_argument("--unconditional_guidance_scale", type=float, default=7.5)
    p.add_argument("--fs",                       type=int, default=10)
    p.add_argument("--seed",                     type=int, default=123)
    p.add_argument("--scale_factor",             type=int, default=8)
    p.add_argument("--clip_model_name",          type=str, default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    args = p.parse_args()
    main(args)
