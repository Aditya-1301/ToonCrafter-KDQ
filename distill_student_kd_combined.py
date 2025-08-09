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
from torchvision.utils import make_grid, save_image
from custom_utils.debugging_utils import debug_tensor


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
        self.lpips = lpips.LPIPS(net="vgg").to(device).eval()
        for p in self.lpips.parameters():
            p.requires_grad_(False)

    def forward(self, student_img, teacher_img):
        return self.l1(student_img, teacher_img), \
               self.lpips(student_img, teacher_img).mean()


# ─────────────────────────── main ───────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--teacher_config", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--teacher_lora_dir", required=True)
    ap.add_argument("--student_config", required=True)
    ap.add_argument("--output_dir", default="kd_run")
    
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=4)

    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_lpips", type=float, default=0.5)
    ap.add_argument("--t_scale", type=float, default=0.9)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--resume", help="path to previous run for resuming")

    ap.add_argument("--log_img_every", type=int, default=1)
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
    teacher = build_model(args.teacher_config, device)
    teacher.load_state_dict(
        torch.load(args.teacher_ckpt, map_location="cpu")["state_dict"],
        strict=False)
    teacher.model.diffusion_model = PeftModel.from_pretrained(
        teacher.model.diffusion_model,
        args.teacher_lora_dir).to(device).eval()
    teacher.eval(); teacher.requires_grad_(False)

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

            # Debug input stats
            if i == 0 and ep == start_ep and args.debug:
                print(f"[DEBUG] Input stats - Start: min={s0.min():.4f}, max={s0.max():.4f}, mean={s0.mean():.4f}")
                print(f"[DEBUG] Input stats - End: min={e2.min():.4f}, max={e2.max():.4f}, mean={e2.mean():.4f}")

            # teacher inference (no-grad)
            with torch.no_grad():
                t_pred = predict_middle(teacher, s0, e2, prm,
                                        clip_v, clip_p, args.t_scale)

            # student forward (gradients kept)
            s_pred = predict_middle(student, s0, e2, prm, clip_v, clip_p, args.t_scale, grad_decode=True)

            l1, lp = loss_fn(s_pred, t_pred)
            loss = args.lambda_l1 * l1 + args.lambda_lpips * lp

            if i == 0 and ep == start_ep:
                print(f"[CHK] student decode requires_grad = {s_pred.requires_grad}")
                if args.debug:
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
        
                s_pred = predict_middle(student, s0, e2, prm, clip_v, clip_p, args.t_scale, grad_decode=True)
        
                n = s_pred.size(0); n_tot += n
                p01, m01 = (s_pred + 1) / 2, (mid_gt + 1) / 2
        
                agg["l1"]    += F.l1_loss(s_pred, mid_gt).item() * n
                lp_batch      = loss_fn.lpips(s_pred, mid_gt).mean()   # ← fixed
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
                tea = predict_middle(teacher, sf, ef, prm[:n_show],
                                     clip_v, clip_p, args.t_scale,
                                     grad_decode=False)          # teacher vis
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