from __future__ import annotations
import argparse, json, random, math, shutil, gc, csv, os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from einops import rearrange
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from peft import PeftModel
import matplotlib.pyplot as plt

import lpips
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from custom_utils.debugging_utils import debug_tensor

from scripts.evaluation.inference import image_guided_synthesis



# ───────────────────────── reproducibility ──────────────────────────
def set_seed(seed: int, deterministic: bool):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic


# ───────────────────────── model helpers ────────────────────────────
def build_model(cfg_path: str, device: str):
    return instantiate_from_config(OmegaConf.load(cfg_path).model).to(device)


def warm_start_student(model: torch.nn.Module, teacher_ckpt: str):
    raw = torch.load(teacher_ckpt, map_location="cpu")["state_dict"]
    # keep = {k: v for k, v in raw.items()
    #         if k.startswith("model.diffusion_model")
    #         and k in model.state_dict()
    #         and v.shape == model.state_dict()[k].shape}
    keep = {k: v for k, v in raw.items()
            if k in model.state_dict()
            and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(keep, strict=False)
    print(f"[INIT] warm-started student with {len(keep)}/{len(raw)} UNet tensors")


def get_clip(device: str):
    repo = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    clip_v = CLIPModel.from_pretrained(repo).vision_model.to(device).eval()
    clip_p = CLIPProcessor.from_pretrained(repo)
    clip_p.feature_extractor.do_normalize = False
    return clip_v, clip_p


@torch.no_grad()
def encode_clip(vit, proc, imgs, device):
    pil = [to_pil_image(((x + 1) * 0.5).clamp(0, 1).cpu()) for x in imgs]
    t = proc(images=pil, return_tensors="pt").to(device)["pixel_values"]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device)[:, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device)[:, None, None]
    t = (t - mean) / (std + 1e-6)
    return vit(pixel_values=t.to(next(vit.parameters()).dtype)).last_hidden_state


# ─────────────────── safe decoder (new!) ─────────────────────────────
def safe_decode_single_frame(vae, z, scale):
    """
    Gradient-tracking VAE decode for a single latent frame.
    Always passes `timesteps=1`, which is what autoencoder_dualref expects.
    """
    return vae.decode(z / scale, timesteps=1)          # ← the crucial one-liner


# ─────────────────── middle-frame predictor ─────────────────────────
def predict_middle(net, start, end, prompts, clip_v, clip_p, t_scale=.8, grad_decode=False):
    B, dev = start.shape[0], start.device
    T, mid = 16, 7

    vid=torch.cat([
        start.unsqueeze(2).repeat(1,1,mid,1,1),
        torch.zeros_like(start).unsqueeze(2),
        end.unsqueeze(2).repeat(1,1,T-mid-1,1,1)],2)

    z=net.encode_first_stage(rearrange(vid,"b c t h w -> (b t) c h w"))
    z = debug_tensor("Latent z (input to UNet)", z, detailed=True)
    z=rearrange(z,"(b t) c h w -> b c t h w",b=B,t=T)
    z_cond=z.clone(); z_cond[:,:,mid]=0

    ctx=torch.cat([
        net.get_learned_conditioning(prompts),
        net.image_proj_model(encode_clip(clip_v,clip_p,start,dev))],1)

    # diffusion timestep (used *only* inside the UNet)
    t_lat=torch.full((B,), int(net.num_timesteps*t_scale), device=dev, dtype=torch.long)

    noise=torch.randn_like(z)
    noisy=net.q_sample(z,t_lat,noise)
    noisy = debug_tensor("Noisy latent (input to UNet)", noisy, detailed=True)
    eps=net.model.diffusion_model(torch.cat([noisy,z_cond],1),t_lat,context=ctx)
    eps = debug_tensor("UNet output eps (predicted noise)", eps, detailed=True)
    z_mid=net.predict_start_from_noise(noisy,t_lat,eps)[:,:,mid]

    # single-frame VAE decode → timesteps must be 1
    # ─── debug / NaN guard ───
    if torch.isnan(z_mid).any():
        print(f"[WARN] NaNs in z_mid at step {t_lat[0].item()}: "
              f"{z_mid.isnan().sum().item()} elements – zeroing out")
        z_mid = torch.nan_to_num(z_mid, nan=0.0, posinf=0.0, neginf=0.0)
        z_mid = debug_tensor("NaN-guarded z_mid (predicted middle frame latent)", z_mid, detailed=True)

    img = safe_decode_single_frame(net.first_stage_model, z_mid, net.scale_factor)
    # img = safe_decode_single_frame(net.first_stage_model, 0.25 * z_mid / net.scale_factor) t_scale=args.t_scale
    img = debug_tensor("Decoded image", img, detailed=True)
    return img if grad_decode else img.clamp(-1, 1)


def predict_middle_latent(net, start, end, prompts, clip_v, clip_p, t_scale=.8):
    B, dev = start.shape[0], start.device
    T, mid = 16, 7

    vid = torch.cat([
        start.unsqueeze(2).repeat(1,1,mid,1,1),
        torch.zeros_like(start).unsqueeze(2),
        end.unsqueeze(2).repeat(1,1,T-mid-1,1,1)], 2)

    # VAE encode (no need to track grads through the VAE encoder)
    with torch.no_grad():
        z = net.encode_first_stage(rearrange(vid, "b c t h w -> (b t) c h w"))
        z = rearrange(z, "(b t) c h w -> b c t h w", b=B, t=T)

    z_cond = z.clone(); z_cond[:, :, mid] = 0

    ctx = torch.cat([
        net.get_learned_conditioning(prompts),
        net.image_proj_model(encode_clip(clip_v, clip_p, start, dev))], 1)

    t_lat = torch.full((B,), int(net.num_timesteps * t_scale),
                       device=dev, dtype=torch.long)
    noise = torch.randn_like(z)
    noisy = net.q_sample(z, t_lat, noise)

    eps = net.model.diffusion_model(torch.cat([noisy, z_cond], 1),
                                    t_lat, context=ctx)
    z_mid = net.predict_start_from_noise(noisy, t_lat, eps)[:, :, mid]

    # Return middle-frame latent and the UNet's eps at the middle frame
    return z_mid, eps[:, :, mid] 


@torch.no_grad()
def generate_teacher_prediction(net, start, end, prompts, ddim_steps=20,
                                unconditional_guidance_scale=7.5):
    B, C, H, W = start.shape
    T = 16
    fs, fe = 0, T - 1
    middle_idx = T // 2 - 1

    videos = torch.cat([
        start.unsqueeze(2).repeat(1, 1, T // 2, 1, 1),
        end.unsqueeze(2).repeat(1, 1, T // 2, 1, 1)
    ], dim=2)  # [B, C, T, H, W]

    noise_shape = [B, net.model.diffusion_model.out_channels, T, H // 8, W // 8]

    batch_samples = image_guided_synthesis(
        net, prompts, videos, noise_shape,
        n_samples=1, ddim_steps=ddim_steps,
        unconditional_guidance_scale=unconditional_guidance_scale,
        interp=True, fs=fs, fe=fe
    )
    # Try to parse outputs defensively:
    # expected: [n_samples, B_ret, C, T, H, W] OR [n_samples, C, T, H, W]
    x = batch_samples[0]  # drop n_samples

    if x.dim() == 5:                # [B_ret, C, T, H, W]
        frames = x
    elif x.dim() == 4:              # [C, T, H, W]  (no batch dim)
        frames = x.unsqueeze(0)     # -> [1, C, T, H, W]
    else:
        raise RuntimeError(f"Unexpected shape from image_guided_synthesis: {x.shape}")

    # If the returned batch is 1, repeat to match B (safe for our use)
    if frames.size(0) != B:
        if frames.size(0) == 1:
            frames = frames.repeat(B, 1, 1, 1, 1)
        else:
            frames = frames[:B]

    teacher_imgs = frames[:, :, middle_idx]  # [B, C, H, W]

    # Ensure range matches the rest of the pipeline ([-1, 1])
    if teacher_imgs.min() >= 0.0 and teacher_imgs.max() <= 1.0:
        teacher_imgs = teacher_imgs * 2.0 - 1.0

    return teacher_imgs


# ─────────────────── state management ─────────────────────────────
def save_training_state(epoch, optimizer, scheduler, scaler, best_loss, file_path):
    state = {'epoch': epoch, 'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
             'scaler_state_dict': scaler.state_dict(), 'best_val_lpips': best_loss}
    torch.save(state, file_path)

def load_training_state(file_path):
    if os.path.exists(file_path):
        return torch.load(file_path, map_location="cpu")
    return None

# ───────────────────────── loss wrapper ─────────────────────────────
class DistillLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = torch.nn.L1Loss()
        self.l1_latent = torch.nn.SmoothL1Loss(beta=0.1)
        self.mse = torch.nn.MSELoss()
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, s_img, t_img, s_z, t_z, s_eps, t_eps):
        # Image-space terms (optional)
        if (s_img is None) or (t_img is None):
            l1_img = torch.tensor(0.0, device=s_z.device)
            lp     = torch.tensor(0.0, device=s_z.device)
        else:
            s = s_img.clamp(-1, 1); t = t_img.clamp(-1, 1)
            l1_img = self.l1(s, t)
            lp = self.lpips(s.float(), t.float()).mean()

        # Latent / eps terms
        lz   = self.l1_latent(s_z, t_z)
        leps = self.mse(s_eps, t_eps)
        return l1_img, lp, lz, leps


# ─────────────────────────── main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--teacher_config", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--teacher_lora_dir", required=False, default=None,
                    help="Path to LoRA weights for the teacher model")
    ap.add_argument("--student_config", required=True)
    ap.add_argument("--output_dir", default="kd_run")
    
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)

    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--lambda_latent", type=float, default=0.5)
    ap.add_argument("--lambda_eps", type=float, default=0.25)

    ap.add_argument("--t_scale", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", help="path to previous run for resuming")

    ap.add_argument("--log_img_every", type=int, default=1)
    ap.add_argument("--lora_scale", type=float, default=1.0, help="LoRA scaling factor for teacher (if LoRA is used)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed, args.deterministic)

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out / "tb")
    
    # Better CSV handling
    csv_path = out / "metrics.csv"
    if not args.resume or not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch', 'l1', 'lpips', 'psnr', 'ssim'])

    # ─────────────────── build teacher (frozen) ────────────────────

    teacher = build_model(args.teacher_config, "cpu") # Load to CPU first
    teacher.load_state_dict(
        torch.load(args.teacher_ckpt, map_location="cpu")["state_dict"],
        strict=False)
    
    if args.teacher_lora_dir:
        print(f"[INIT] Attaching LoRA to teacher from: {args.teacher_lora_dir}")
        teacher.model.diffusion_model = PeftModel.from_pretrained(
            teacher.model.diffusion_model,
            args.teacher_lora_dir, is_trainable=False) # Ensure it's not trainable
        
        # Apply inference-time scaling
        cfg_file = Path(args.teacher_lora_dir) / "adapter_config.json"
        if not cfg_file.exists(): raise FileNotFoundError(f"{cfg_file} not found.")
        l_cfg   = OmegaConf.load(cfg_file)
        l_alpha = l_cfg.get("lora_alpha", 16)
        for mod in teacher.model.diffusion_model.modules():
            if hasattr(mod, "lora_A") and hasattr(mod, "r"):
                rank = mod.r['default'] if isinstance(mod.r, dict) else mod.r
                scaling_value = l_alpha / rank
                scaling_value *= args.lora_scale # Use the command-line argument
                mod.scaling = {'default': scaling_value}
        print(f"    ✔ LoRA scaling set to {args.lora_scale}")
    
    teacher = teacher.to(device)
    teacher.eval()
    teacher.requires_grad_(False)
    print("    ✔ Teacher model finalized on GPU and frozen.")

    # ─────────────────── build student ─────────────────────────────
    student = build_model(args.student_config, device)
    warm_start_student(student, args.teacher_ckpt)
    # ------------------------------------------------------------------
    student.first_stage_model.load_state_dict(teacher.first_stage_model.state_dict())
    student.first_stage_model.requires_grad_(False)
    # ------------------------------------------------------------------

    student.train()

    clip_v, clip_p = get_clip(device)
    loss_fn = DistillLoss(device)
    psnr_m = PeakSignalNoiseRatio(data_range=1.).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.).to(device)

    # ─────────────────── dataset & split ───────────────────────────
    ds = ATD12K_Dataset(args.dataset_path,
                        video_size=(320, 512),
                        split="train")
    split_file = out / "split.json"
    if args.resume and split_file.exists():
        idx = json.load(split_file.open())
        tr_idx, va_idx = idx["train"], idx["val"]
    else:
        ids = list(range(len(ds))); random.shuffle(ids)
        v = int(0.1 * len(ids))
        tr_idx, va_idx = ids[v:], ids[:v]
        json.dump({"train": tr_idx, "val": va_idx}, split_file.open("w"))

    dl_kw = dict(batch_size=args.bs,
                 num_workers=4,
                 pin_memory=True,
                 persistent_workers=False)
    train_dl = DataLoader(Subset(ds, tr_idx), shuffle=True, **dl_kw)
    val_dl = DataLoader(Subset(ds, va_idx), shuffle=False, **dl_kw)

    if args.debug:                               # cut to a single batch each
        train_dl = DataLoader(Subset(ds, tr_idx[:args.bs]),
                              shuffle=False, **dl_kw)
        val_dl = DataLoader(Subset(ds, va_idx[:args.bs]),
                            shuffle=False, **dl_kw)

    print(f"[INIT] Datasets: Train={len(train_dl.dataset)}  "
          f"Val={len(val_dl.dataset)}")

    # ─────────────────── optim & sched ─────────────────────────────
    opt = AdamW(student.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = CosineAnnealingLR(opt,
                              T_max=args.epochs * len(train_dl) // args.accum)
    scaler = GradScaler()

    # ─────────────────── resume (optional) ─────────────────────────
    start_ep, best_lpips = 1, math.inf
    latest_dir = out / "latest"
    if args.resume and latest_dir.exists():
        state_path = latest_dir / "state.pt"
        ckpt_path = latest_dir / "student.ckpt"
        if state_path.exists() and ckpt_path.exists():
            print(f"[RESUME] Loading checkpoint from {args.resume}")
            student.load_state_dict(torch.load(ckpt_path)["state_dict"])
            state = load_training_state(state_path)
            if state:
                opt.load_state_dict(state['optimizer_state_dict'])
                if state.get('scheduler_state_dict'):
                    sched.load_state_dict(state['scheduler_state_dict'])
                scaler.load_state_dict(state['scaler_state_dict'])
                start_ep = state.get('epoch', 0) + 1
                best_lpips = state.get('best_val_lpips', math.inf)
                print(f"[RESUME] State loaded. Resuming from epoch {start_ep}")

    # ─────────────────── training loop ─────────────────────────────
    global_step = (start_ep - 1) * len(train_dl)
    for ep in range(start_ep, args.epochs + 1):
        student.train()
        for i, batch in enumerate(tqdm(train_dl, desc=f"train {ep}")):
            s0 = batch["start_frame"].to(device)
            e2 = batch["end_frame"].to(device)
            prm = batch.get("prompt", [""] * s0.size(0))
            need_img_loss = (args.lambda_l1 > 0 or args.lambda_lpips > 0)

            # Debug input stats
            if i == 0 and ep == start_ep and args.debug:
                print(f"[DEBUG] Input stats - Start: min={s0.min():.4f}, max={s0.max():.4f}, mean={s0.mean():.4f}")
                print(f"[DEBUG] Input stats - End: min={e2.min():.4f}, max={e2.max():.4f}, mean={e2.mean():.4f}")

            # --- teacher (no grad): get latent + eps, then decode for visualization/loss ---
            with torch.no_grad():
                if args.teacher_lora_dir is None:
                    t_pred = generate_teacher_prediction(teacher, s0, e2, prm, ddim_steps=20).clamp(-1, 1) if need_img_loss else None
                    t_z, t_eps = None, None
                else:
                    t_z, t_eps = predict_middle_latent(teacher, s0, e2, prm, clip_v, clip_p, args.t_scale)
                    t_pred = safe_decode_single_frame(teacher.first_stage_model, t_z, teacher.scale_factor).clamp(-1, 1) if need_img_loss else None
            
            s_z, s_eps = predict_middle_latent(student, s0, e2, prm, clip_v, clip_p, args.t_scale)
            s_pred = (safe_decode_single_frame(student.first_stage_model, s_z, student.scale_factor)
                          if need_img_loss else None)
            
            l1_img, lp, lz, leps = loss_fn(s_pred, t_pred,
                                           s_z, (t_z if t_z is not None else s_z.detach()*0),
                                           s_eps, (t_eps if t_eps is not None else s_eps.detach()*0))

            w_lat = 0.0 if t_z is None else args.lambda_latent
            w_eps = 0.0 if t_eps is None else args.lambda_eps
            
            loss = (args.lambda_l1 * l1_img
                    + args.lambda_lpips * lp
                    + w_lat * lz
                    + w_eps * leps)

            if i == 0 and ep == start_ep:
                print(f"[CHK] student decode requires_grad = {getattr(s_pred, 'requires_grad', False)}")
                if args.debug and need_img_loss:
                    print(f"[DEBUG] Teacher output stats - Min: {t_pred.min():.4f}, Max: {t_pred.max():.4f}, Mean: {t_pred.mean():.4f}")
                    print(f"[DEBUG] Student output stats - Min: {s_pred.min():.4f}, Max: {s_pred.max():.4f}, Mean: {s_pred.mean():.4f}")
            
            scaler.scale(loss / args.accum).backward()

            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                if i == 0 and ep == start_ep:
                    gtot = sum(p.grad.abs().sum()
                               for p in student.parameters() if p.grad is not None).item()
                    print(f"[CHK] raw grad sum = {gtot:.6e}") 
                
                # Better gradient clipping with warnings
                grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                if grad_norm > 10.0:
                    print(f"[WARN] Large grad norm {grad_norm:.2f}")
                
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True); sched.step()
                writer.add_scalar("train/loss", loss.item(), global_step)
                global_step += 1

        # ───── validation ─────
        student.eval(); agg = defaultdict(float); n_tot = 0
        with torch.no_grad():
            for vb in tqdm(val_dl, desc=f"val {ep}"):
                s0, mid_gt, e2 = (vb["start_frame"].to(device),
                                  vb["ground_truth_middle"].to(device),
                                  vb["end_frame"].to(device))
                prm = vb.get("prompt", [""] * s0.size(0))
        
                # s_pred = predict_middle(student, s0, e2, prm, clip_v, clip_p, args.t_scale, grad_decode=True)
                s_z, s_eps = predict_middle_latent(student, s0, e2, prm, clip_v, clip_p, args.t_scale)
                s_pred = safe_decode_single_frame(student.first_stage_model, s_z, student.scale_factor)
        
                n = s_pred.size(0); n_tot += n
                p01, m01 = (s_pred + 1) / 2, (mid_gt + 1) / 2
        
                agg["l1"]    += F.l1_loss(s_pred, mid_gt).item() * n
                # lp_batch      = loss_fn.lpips(s_pred, mid_gt).mean()   # ← fixed
                lp_batch      = loss_fn.lpips(s_pred.clamp(-1,1), mid_gt.clamp(-1,1)).mean()
                agg["lpips"] += lp_batch.item() * n
                agg["psnr"]  += psnr_m(p01, m01).item() * n
                agg["ssim"]  += ssim_m(p01, m01).item() * n
        stats = {k: v / n_tot for k, v in agg.items()} if n_tot > 0 else {}

        if n_tot == 0:
            print(f"[E{ep}] WARNING: No validation batches processed.")
        else:
            print(f"[E{ep}] L1 {stats['l1']:.4f}  "
                  f"LPIPS {stats['lpips']:.4f}  "
                  f"PSNR {stats['psnr']:.2f}  "
                  f"SSIM {stats['ssim']:.4f}")

        # TB + CSV
        for k, v in stats.items():
            writer.add_scalar(f"val/{k}", v, ep)
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, stats.get('l1', 0), stats.get('lpips', 0),
                                   stats.get('psnr', 0), stats.get('ssim', 0)])

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        img_grid_dir = out / "checkpoint_images"
        img_grid_dir.mkdir(exist_ok=True)


        # ───────────── save PNG image-grid ─────────────
        if ep % 1 == 0:                                   # change 1→args.log_img_every if you add a flag
            n_show = min(4, s0.size(0))                   # 4 samples → 4 columns

            with torch.no_grad():
                sf = s0[:n_show]
                ef = e2[:n_show]
                gt = mid_gt[:n_show]
                if args.teacher_lora_dir is None:
                    tea = generate_teacher_prediction(teacher, sf, ef, prm[:n_show], ddim_steps=20).clamp(-1,1)
                else:
                    tea = predict_middle(teacher, sf, ef, prm[:n_show],
                                         clip_v, clip_p, args.t_scale,
                                         grad_decode=False)       # teacher vis
                stu = predict_middle(student, sf, ef, prm[:n_show],
                                     clip_v, clip_p, args.t_scale,
                                     grad_decode=False)           # student vis
            
            # TensorBoard grid: Teacher, Student, GT  (three rows)
            grid_tb = make_grid(torch.cat([(tea + 1) / 2,
                                           (stu + 1) / 2,
                                           (gt  + 1) / 2], 0),
                                nrow=n_show)
            writer.add_image("val/images", grid_tb, ep)
            
            # PNG grid: Start | GT | Teacher-Pred | Student-Pred | End
            titles = ["Start", "Ground Truth", "Teacher-Pred", "Student-Pred", "End"]
            fig, axes = plt.subplots(n_show, 5, figsize=(20, 4 * n_show), squeeze=False)
            for i in range(n_show):
                for j, (img, title) in enumerate(zip([sf[i], gt[i], tea[i], stu[i], ef[i]], titles)):
                    # 1) clamp into [-1,1]
                    img_vis = img.clamp(-1.0, 1.0)
            
                    # 2) map [-1,1] → [0,1]
                    img_vis = (img_vis + 1.0) * 0.5
                    # arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
                    arr = (img_vis.permute(1,2,0).cpu().numpy() * 255.0).round().clip(0,255).astype("uint8")
                    axes[i, j].imshow(arr)
                    axes[i, j].set_title(f"{title}\n(E{ep})")
                    axes[i, j].axis("off")
            plt.tight_layout()
            fig.savefig(img_grid_dir / f"epoch_{ep:04d}_grid.png", dpi=150)
            plt.close(fig)

        # ───────── checkpoint ─────────
        latest_dir.mkdir(exist_ok=True)
        torch.save({"state_dict": student.state_dict()},
                   latest_dir / "student.ckpt")
        save_training_state(ep, opt, sched, scaler, best_lpips, latest_dir / "state.pt")
        shutil.copy(split_file, latest_dir / "split.json")

        if stats.get("lpips", math.inf) < best_lpips:
            best_lpips = stats["lpips"]
            print(f"✨ New best val LPIPS: {best_lpips:.4f}. Saving best checkpoint.")
            shutil.copytree(latest_dir, out / "best", dirs_exist_ok=True)

        if args.debug:
            print("[DEBUG] debug run complete – exiting after one epoch")
            break

        gc.collect(); torch.cuda.empty_cache()

    writer.close()
    print("✓ Knowledge-distillation complete")


if __name__ == "__main__":
    main()