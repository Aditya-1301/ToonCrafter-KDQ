import os, csv, time, argparse, random, shutil, multiprocessing, traceback
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from pytorch_lightning import seed_everything
from einops import rearrange
from transformers import CLIPModel, CLIPProcessor
from torchvision.transforms.functional import to_pil_image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Local Imports from your project structure ---
from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset


# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def parse_args():
    """Parses command-line arguments."""
    p = argparse.ArgumentParser(description="LoRA fine-tuning for ToonCrafter.")
    p.add_argument("--dataset_path", type=str, required=True); p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--config", type=str, required=True); p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_rank", type=int, default=16); p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lambda_latent",type=float, default=1.0); p.add_argument("--lambda_pixel", type=float, default=1.0)
    p.add_argument("--val_frac", type=float, default=0.1); p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512); p.add_argument("--log_img_every", type=int, default=1)
    p.add_argument("--debug", action="store_true"); p.add_argument("--use_scheduler", action="store_true", help="Enable CosineAnnealingLR scheduler.")
    p.add_argument("--lambda_flow", type=float, default=0.5,
               help="Weight for optical-flow warp loss.")
    p.add_argument("--flow_ckpt", type=str,
                   default="checkpoints/raft-small.pth",
                   help="Pre-trained RAFT weights.")
    p.add_argument("--resume_dir", type=str, help="Path to a previous run's output directory to resume from.")
    return p.parse_args()

def load_checkpoint(model, ckpt_path):
    """Loads weights from a checkpoint file into the model."""
    print(f"[INIT] Loading base model checkpoint: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu"); sd = sd.get("state_dict", sd)
    new_sd = {k.replace("model.model.diffusion_model.", "model.diffusion_model.") if k.startswith("model.model.diffusion_model.") else k: v for k, v in sd.items()}
    model.load_state_dict(new_sd, strict=False)
    print("[INIT] Checkpoint loaded successfully (non-strict).")
    return model

def get_unet(model):
    """Safely retrieves the UNet from the main model object."""
    return model.model.diffusion_model

@torch.no_grad()
def get_latent_z(model, vids):
    """Encodes a batch of videos into the latent space."""
    b, c, t, h, w = vids.shape
    flat = rearrange(vids, "b c t h w -> (b t) c h w")
    z = model.encode_first_stage(flat)
    return rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)

@torch.no_grad()
def compute_metrics(gt, pred):
    """Computes PSNR and SSIM with robust handling of NaN/inf values."""
    gt_np = gt.cpu().numpy().transpose(0, 2, 3, 1)
    pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
    psnrs, ssims = [], []

    for g, p in zip(gt_np, pred_np):
        # Clamp infinite PSNR to a high, finite value
        psnr_val = peak_signal_noise_ratio(g, p, data_range=1.0)
        psnrs.append(psnr_val if np.isfinite(psnr_val) else 100.0)

        # Ensure SSIM window is valid and handle NaNs from flat regions
        win = min(7, g.shape[0], g.shape[1])
        if win % 2 == 0: win -= 1
        if win >= 3:
            ssim_val = ssim(g, p, data_range=1.0, channel_axis=2, win_size=win) 
            ssims.append(ssim_val if np.isfinite(ssim_val) else 0.0)
        else:
            ssims.append(0.0)
            
    return float(np.mean(psnrs)), float(np.mean(ssims))

def get_clip_embedder(device):
    """Initializes and returns the CLIP vision model and processor."""
    name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    print(f"[INIT] Loading CLIP vision model: {name}")
    clip_model = CLIPModel.from_pretrained(name).vision_model.to(device)
    processor = CLIPProcessor.from_pretrained(name)
    return clip_model, processor, clip_model.config.hidden_size

def encode_clip(clip_model, processor, imgs, device, adapter=None):
    """Encodes images with CLIP, with robust handling of data types and non-finite values."""
    pil = [to_pil_image((x*0.5+0.5).clamp(0,1).cpu()) for x in imgs]
    inputs = processor(images=pil, return_tensors="pt", do_rescale=False).to(device)
    
    pixel_values = inputs["pixel_values"]
    if not torch.all(torch.isfinite(pixel_values)):
        print("[WARN] Non-finite values detected in CLIP input. Sanitizing.")
        pixel_values = torch.nan_to_num(pixel_values, nan=0.0, posinf=1.0, neginf=-1.0)

    with torch.no_grad():
        target_dtype = next(clip_model.parameters()).dtype
        features = clip_model(pixel_values=pixel_values.to(target_dtype)).last_hidden_state
        
    return adapter(features) if adapter is not None else features

def save_training_state(epoch: int, optimizer, scheduler, best_lpips: float, file_path: str):
    """Saves the complete training state to a file."""
    state = {
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_lpips': best_lpips
    }
    torch.save(state, file_path)

def load_training_state(file_path: str):
    """Loads a training state from a file."""
    if os.path.exists(file_path):
        return torch.load(file_path, map_location="cpu")
    return None

# =================================================================================================
# MAIN TRAINING SCRIPT
# =================================================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INIT] Output dir: {args.output_dir}")

    seed_everything(args.seed, workers=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Using device: {device}")

    writer = SummaryWriter(os.path.join(args.output_dir, "tb"))
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_latent", "train_lpips", "train_psnr", "train_ssim",
                                "val_latent", "val_lpips", "val_psnr", "val_ssim", "lr"])

    if args.debug:
        print("[DEBUG] Anomaly detection enabled.")
        torch.autograd.set_detect_anomaly(True)

    cfg = OmegaConf.load(args.config)
    model = instantiate_from_config(cfg.model); model = load_checkpoint(model, args.ckpt_path).to(device)
    unet = get_unet(model)
    for param in unet.parameters(): param.requires_grad = False

    # lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=["to_q","to_k","to_v","to_out.0"], lora_dropout=0.1, bias="none")
    # lora_unet = get_peft_model(unet, lora_config)

    # --- THIS IS THE FINAL, CORRECTED BLOCK ---
    target_modules = []
    # Iterate through all modules in the UNet
    for name, _ in unet.named_modules():
        # We are looking for the linear layers (to_q, to_k, etc.) that are
        # specifically inside a spatial cross-attention block (attn2).
        if "attn2" in name and name.endswith((".to_q", ".to_k", ".to_v", ".to_out.0")):
            target_modules.append(name)
    
    print(f"\n[INIT] Found {len(target_modules)} target linear layers for SPATIAL LoRA:")
    for name in sorted(target_modules):
        print(f"  ↪ {name}")
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules, # Use the dynamically found linear layer names
        lora_dropout=0.1,
        bias="none"
    )
    # --- END OF CORRECTED BLOCK ---
    
    lora_unet = get_peft_model(unet, lora_config)
    
    trainable_params = list(filter(lambda p: p.requires_grad, lora_unet.parameters()))
    lora_unet.print_trainable_parameters()

    clip_model, clip_proc, clip_dim = get_clip_embedder(device)
    resampler_proj_in = model.image_proj_model.proj_in
    adapter = None
    if clip_dim != resampler_proj_in.in_features:
        print(f"[INIT] Creating adapter for CLIP features: {clip_dim} -> {resampler_proj_in.in_features}")
        adapter = torch.nn.Linear(clip_dim, resampler_proj_in.in_features).to(device)
        trainable_params.extend(list(adapter.parameters()))
    print(f"[INIT] Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    full_dataset = ATD12K_Dataset(args.dataset_path, video_size=(args.height, args.width), split="train")
    if args.debug:
        indices = list(range(16)); train_dataset, val_dataset = Subset(full_dataset, indices[:8]), Subset(full_dataset, indices[8:])
    else:
        n_val = int(len(full_dataset)*args.val_frac); n_train = len(full_dataset)-n_val
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))
    print(f"[INIT] Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}")

    dl_kwargs = dict(batch_size=args.bs, shuffle=not args.debug, num_workers=4, pin_memory=True, persistent_workers=True)
    train_dataloader, val_dataloader = DataLoader(train_dataset, **dl_kwargs), DataLoader(val_dataset, **dl_kwargs)

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    
    if scheduler: print("[INIT] Using CosineAnnealingLR scheduler.")

    start_epoch = 1
    best_val_lpips = float("inf")
    if args.resume_dir:
        print(f"[RESUME] Attempting to resume from directory: {args.resume_dir}")
        lora_weights_path = os.path.join(args.resume_dir, "lora_latest")
        state_path = os.path.join(args.resume_dir, "state_latest.pt")

        if not os.path.isdir(lora_weights_path):
            raise FileNotFoundError(f"LoRA directory not found at: {lora_weights_path}")

        # Load LoRA weights from the checkpoint
        lora_unet.load_adapter(lora_weights_path, "default")
        print("[RESUME] LoRA adapter weights loaded.")

        if adapter:
            adapter_path = os.path.join(lora_weights_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                adapter.load_state_dict(torch.load(adapter_path, map_location=device))
                print("[RESUME] CLIP adapter weights loaded.")

        # Load training state bundle
        training_state = load_training_state(state_path)
        if training_state:
            start_epoch = training_state['epoch'] + 1
            best_val_lpips = training_state['best_val_lpips']
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in training_state and training_state['scheduler_state_dict'] is not None:
                scheduler.load_state_dict(training_state['scheduler_state_dict'])
            print(f"[RESUME] Optimizer and scheduler states loaded. Resuming from epoch {start_epoch}.")
        else:
            print("[WARN] Could not find state_latest.pt. Resuming weights only, optimizer and scheduler are reset.")

        lora_unet.set_adapter("default")

    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    img_grid_dir = os.path.join(args.output_dir, "checkpt_images"); os.makedirs(img_grid_dir, exist_ok=True)
    
    print("\n" + "="*40 + "\n>>> STARTING TRAINING <<<\n" + "="*40 + "\n")

    for epoch in range(start_epoch, args.epochs + 1):
        lora_unet.train()
        if adapter: adapter.train()
        train_metrics = {"latent": 0.0, "lpips": 0.0, "psnr": 0.0, "ssim": 0.0}

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"[Epoch {epoch}] Train")):
            if args.debug and step >= 5: break
            try:
                start, middle_gt, end = batch["start_frame"].to(device), batch["ground_truth_middle"].to(device), batch["end_frame"].to(device)
                B = start.shape[0]

                with torch.no_grad(): z_all = get_latent_z(model, torch.stack([start, middle_gt, end], dim=2))
                x_start, c_concat = z_all.clone(), z_all.clone()
                c_concat[:, :, 1, :, :].zero_()

                prompts = batch.get("prompt", [""] * B)
                context = torch.cat([model.get_learned_conditioning(prompts), model.image_proj_model(encode_clip(clip_model, clip_proc, start, device, adapter))], dim=1)

                t, noise = torch.randint(0, model.num_timesteps, (B,), device=device), torch.randn_like(x_start)
                noisy = model.q_sample(x_start=x_start, t=t, noise=noise)

                unet_input = torch.cat([noisy, c_concat], dim=1)
                assert unet_input.shape[1] == unet.in_channels, f"UNet channel mismatch"
                pred_noise = lora_unet(unet_input, t, context=context)

                loss_latent = F.l1_loss(pred_noise, noise)

                z_pred = model.predict_start_from_noise(noisy, t, pred_noise)[:, :, 1, :, :]
                with torch.no_grad(): img_gt = model.decode_first_stage(x_start[:, :, 1, :, :])
                img_pred = model.decode_first_stage(z_pred)

                loss_pixel = lpips_fn(img_pred, img_gt).mean()
                if not torch.isfinite(loss_pixel):
                    print(f"[WARN] NaN LPIPS at E{epoch} S{step}. Clamping to 0."); loss_pixel = torch.tensor(0.0, device=device)

                total_loss = (args.lambda_latent * loss_latent) + (args.lambda_pixel * loss_pixel)
                optimizer.zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(trainable_params, 1.0); optimizer.step()

                psnr, ssim_val = compute_metrics((img_gt/2+0.5).clamp(0,1), (img_pred/2+0.5).clamp(0,1))
                train_metrics["latent"]+=loss_latent.item(); train_metrics["lpips"]+=loss_pixel.item(); train_metrics["psnr"]+=psnr; train_metrics["ssim"]+=ssim_val
            except Exception as e:
                print(f"\n[ERROR] Step {step}, Epoch {epoch}: {e}\n{traceback.format_exc()}"); continue

        num_steps = (min(len(train_dataloader), 5) if args.debug else len(train_dataloader)) or 1
        for k in train_metrics: train_metrics[k] /= num_steps

        lora_unet.eval()
        if adapter: adapter.eval()
        val_metrics = {"latent": 0.0, "lpips": 0.0, "psnr": 0.0, "ssim": 0.0}
        grid_tensors = None

        with torch.no_grad():
            for step, batch in enumerate(tqdm(val_dataloader, desc=f"[Epoch {epoch}] Val")):
                if args.debug and step >= 5: break
                start, middle_gt, end = batch["start_frame"].to(device), batch["ground_truth_middle"].to(device), batch["end_frame"].to(device)
                B = start.shape[0]

                z_all = get_latent_z(model, torch.stack([start, middle_gt, end], dim=2))
                x_start, c_concat = z_all.clone(), z_all.clone(); c_concat[:, :, 1, :, :].zero_()
                prompts = batch.get("prompt", [""] * B)
                context = torch.cat([model.get_learned_conditioning(prompts), model.image_proj_model(encode_clip(clip_model, clip_proc, start, device, adapter))], dim=1)

                t, noise = torch.randint(0, model.num_timesteps, (B,), device=device), torch.randn_like(x_start)
                noisy = model.q_sample(x_start=x_start, t=t, noise=noise)
                pred_noise = lora_unet(torch.cat([noisy, c_concat], dim=1), t, context=context)

                loss_latent = F.l1_loss(pred_noise, noise)
                z_pred = model.predict_start_from_noise(noisy, t, pred_noise)[:, :, 1, :, :]
                img_pred, img_gt = model.decode_first_stage(z_pred), model.decode_first_stage(x_start[:,:,1,:,:])

                loss_pixel = lpips_fn(img_pred, img_gt).mean()
                if not torch.isfinite(loss_pixel): loss_pixel = torch.tensor(0.0)

                psnr, ssim_val = compute_metrics((img_gt/2+0.5).clamp(0,1), (img_pred/2+0.5).clamp(0,1))
                val_metrics["latent"]+=loss_latent.item(); val_metrics["lpips"]+=loss_pixel.item(); val_metrics["psnr"]+=psnr; val_metrics["ssim"]+=ssim_val

                if step == 0: grid_tensors = (start.cpu(), (img_gt/2+0.5).clamp(0,1).cpu(), (img_pred/2+0.5).clamp(0,1).cpu(), end.cpu())

        num_steps = (min(len(val_dataloader), 5) if args.debug else len(val_dataloader)) or 1
        for k in val_metrics: val_metrics[k] /= num_steps

        print(f"\n[Epoch {epoch}] Train -> L1: {train_metrics['latent']:.4f}, LPIPS: {train_metrics['lpips']:.4f} | Val -> L1: {val_metrics['latent']:.4f}, LPIPS: {val_metrics['lpips']:.4f}")
        
        lr_current = scheduler.get_last_lr()[0] if scheduler else args.lr
        writer.add_scalar("Loss/train_latent", train_metrics['latent'], epoch)
        writer.add_scalar("Loss/train_lpips", train_metrics['lpips'], epoch)
        writer.add_scalar("Metrics/train_psnr", train_metrics['psnr'], epoch)
        writer.add_scalar("Metrics/train_ssim", train_metrics['ssim'], epoch)
        writer.add_scalar("Loss/val_latent", val_metrics['latent'], epoch)
        writer.add_scalar("Loss/val_lpips", val_metrics['lpips'], epoch)
        writer.add_scalar("Metrics/val_psnr", val_metrics['psnr'], epoch)
        writer.add_scalar("Metrics/val_ssim", val_metrics['ssim'], epoch)
        writer.add_scalar("Learning_Rate", lr_current, epoch)
        writer.flush()

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_metrics['latent'], train_metrics['lpips'], train_metrics['psnr'], train_metrics['ssim'],val_metrics['latent'], val_metrics['lpips'], val_metrics['psnr'], val_metrics['ssim'], lr_current])
        
        if epoch % args.log_img_every == 0 and grid_tensors is not None:
            sf, gt, pr, ef = grid_tensors
            n_show = min(4, sf.shape[0])
            titles = ["Start", "Ground Truth", "Prediction", "End"]
            fig, axes = plt.subplots(n_show, 4, figsize=(16, 4*n_show), squeeze=False)
            for i in range(n_show):
                for j, (img, title) in enumerate(zip([sf[i], gt[i], pr[i], ef[i]], titles)):
                    if img.min() < -0.1:
                        img = (img * 0.5 + 0.5).clamp(0, 1)
                    arr = (img.permute(1,2,0).cpu().numpy() * 255).round().astype("uint8")
                    axes[i,j].imshow(arr)
                    axes[i,j].set_title(f"{title}\n(E{epoch})")
                    axes[i,j].axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(img_grid_dir, f"epoch_{epoch:04d}_grid.png"), dpi=150)
            plt.close(fig)

        latest_dir, best_dir = os.path.join(args.output_dir, "lora_latest"), os.path.join(args.output_dir, "lora_best")
        lora_unet.save_pretrained(latest_dir)
        if adapter: torch.save(adapter.state_dict(), os.path.join(latest_dir, "adapter_model.bin"))

        save_training_state(epoch, optimizer, scheduler, best_val_lpips, os.path.join(args.output_dir, "state_latest.pt"))
                                                                  
        if val_metrics['lpips'] < best_val_lpips:
            best_val_lpips = val_metrics['lpips']
            print(f"\n[Epoch {epoch}] ✨ New best LPIPS: {best_val_lpips:.4f}. Saving to '{best_dir}'.\n")
            if os.path.exists(best_dir): shutil.rmtree(best_dir)
            shutil.copytree(latest_dir, best_dir)
            shutil.copy2(os.path.join(args.output_dir, "state_latest.pt"), os.path.join(args.output_dir, "state_best.pt"))
        
        if scheduler: scheduler.step()

    writer.close()
    print("\n" + "="*40 + "\n✅ TRAINING COMPLETE ✅\n" + "="*40 + "\n")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()