# File: check_vae.py
import argparse
import os
import sys
from pathlib import Path

import lpips
import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.metrics import (peak_signal_noise_ratio as psnr,
                             structural_similarity as ssim)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add project root to the Python path to find custom_utils
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from custom_utils.datasets import ATD12K_Dataset
from utils.utils import instantiate_from_config

def main(cfg: argparse.Namespace) -> None:
    """
    Performs a VAE reconstruction test on the ATD-12K dataset to measure
    the autoencoder's baseline performance on this specific data distribution.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # --- Load Model ---
    model_conf = OmegaConf.load(cfg.config)
    model = instantiate_from_config(model_conf.model).to(device)
    state = torch.load(cfg.ckpt_path, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state), strict=False)
    model.eval()
    print("[INFO] ToonCrafter model loaded successfully.")

    # --- Load LPIPS Metric ---
    # Using VGG for consistency with your other evaluation scripts.
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    print("[INFO] LPIPS metric (VGG backbone) loaded.")

    # --- Load Data using your custom ATD12K_Dataset class ---
    dataset = ATD12K_Dataset(
        root_dir=cfg.dataset_path,
        video_size=(cfg.height, cfg.width),
        split=cfg.split
    )
    if cfg.debug:
        # For a quick test, only use a small subset of the data
        dataset = Subset(dataset, range(10))
    
    dataloader = DataLoader(dataset, batch_size=cfg.bs, shuffle=False, num_workers=4)
    print(f"[INFO] Evaluating on {len(dataset)} samples from the '{cfg.split}' split.")

    metrics = {"psnr": [], "ssim": [], "lpips": []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating VAE on '{cfg.split}' split"):
            # Aggregate all frames from the triplet (start, middle, end) into one large batch
            original_images = torch.cat([
                batch['start_frame'],
                batch['ground_truth_middle'],
                batch['end_frame']
            ], dim=0).to(device)

            # Perform the VAE round-trip: encode to latent, then decode back to pixel space
            latents = model.encode_first_stage(original_images)
            reconstructions = model.decode_first_stage(latents)

            # Denormalize images from [-1, 1] to [0, 1] for metric calculation
            orig_0_1 = (original_images * 0.5 + 0.5).clamp(0, 1)
            recon_0_1 = (reconstructions * 0.5 + 0.5).clamp(0, 1)

            # Calculate LPIPS for the batch
            lpips_val = lpips_fn(orig_0_1, recon_0_1).flatten().cpu().numpy()
            metrics["lpips"].extend(lpips_val)

            # Calculate PSNR/SSIM on the CPU for each image in the batch
            for i in range(orig_0_1.size(0)):
                orig_np = orig_0_1[i].permute(1, 2, 0).cpu().numpy()
                recon_np = recon_0_1[i].permute(1, 2, 0).cpu().numpy()
                
                # Use a valid window size for SSIM
                win_size = min(7, orig_np.shape[0], orig_np.shape[1])
                if win_size % 2 == 0: win_size -= 1

                metrics["psnr"].append(psnr(orig_np, recon_np, data_range=1.0))
                metrics["ssim"].append(ssim(orig_np, recon_np, data_range=1.0, channel_axis=2, win_size=win_size))

    # --- Print Final Results ---
    print("\n" + "="*40)
    print(f"  VAE RECONSTRUCTION RESULTS for '{cfg.split}' split")
    print(f"  Average PSNR : {np.mean(metrics['psnr']):.4f} (Higher is better)")
    print(f"  Average SSIM : {np.mean(metrics['ssim']):.4f} (Higher is better)")
    print(f"  Average LPIPS: {np.mean(metrics['lpips']):.4f} (Lower is better)")
    print("="*40 + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate the VAE reconstruction quality of the ToonCrafter model on the ATD-12K dataset.")
    p.add_argument("--config", type=str, required=True, help="Path to the ToonCrafter model config file.")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to the ToonCrafter model checkpoint.")
    p.add_argument("--dataset_path", type=str, required=True, help="Root path of the ATD-12K dataset.")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split to evaluate on.")
    p.add_argument("--height", type=int, default=320, help="Image height for resizing.")
    p.add_argument("--width", type=int, default=512, help="Image width for resizing.")
    p.add_argument("--bs", type=int, default=4, help="Batch size for evaluation.")
    p.add_argument("--debug", action="store_true", help="Run in debug mode on a small subset of data.")
    return p

if __name__ == "__main__":
    main(build_parser().parse_args())