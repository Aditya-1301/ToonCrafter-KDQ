import argparse, os, sys, time, csv
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
import torchvision
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Subset, Dataset
from peft import PeftModel
from PIL import Image
import random

# Add project root to path to solve import issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset
# We use the robust, fast image_guided_synthesis from your best baseline script
from metric_evaluation.evaluate_baseline_metrics import image_guided_synthesis

# --- Helper Classes & Functions ---

class CustomFramesDataset(Dataset):
    """A simple dataset to load start/end frames from a folder."""
    def __init__(self, folder_path, size=(320, 512)):
        self.folder_path = folder_path
        self.start_frame_path = os.path.join(folder_path, 'frame_000.png')
        self.end_frame_path = os.path.join(folder_path, 'frame_001.png')
        if not os.path.exists(self.start_frame_path) or not os.path.exists(self.end_frame_path):
            raise FileNotFoundError(f"Custom folder must contain 'frame_000.png' (start) and 'frame_001.png' (end).")
        self.size = size

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # PIL's resize method expects (width, height), but our size tuple is (height, width).
        # We swap them here to ensure correct resizing.
        correct_size = (self.size[1], self.size[0])
        start_img = Image.open(self.start_frame_path).convert("RGB").resize(correct_size)
        end_img = Image.open(self.end_frame_path).convert("RGB").resize(correct_size)
        
        start_tensor = torchvision.transforms.ToTensor()(start_img) * 2.0 - 1.0
        end_tensor = torchvision.transforms.ToTensor()(end_img) * 2.0 - 1.0
        return {"start_frame": start_tensor, "end_frame": end_tensor, "prompt": "A high motion animation frame"}

def log_performance_metrics(output_dir, model_name, checkpoint_size_gb, peak_vram_gb, avg_ms_per_frame):
    """Logs performance metrics to a dedicated CSV file."""
    log_path = os.path.join(output_dir, "performance_summary.csv")
    is_new_file = not os.path.exists(log_path)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(["Model", "Checkpoint Size (GB)", "Peak VRAM (GB)", "Inference Speed (ms/frame)"])
        writer.writerow([model_name, f"{checkpoint_size_gb:.3f}", f"{peak_vram_gb:.2f}", f"{avg_ms_per_frame:.2f}"])
    print(f"\n[PERF] Performance metrics for '{model_name}' saved to {log_path}")

def save_video(video_tensor, filename, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    video = video_tensor.detach().cpu().float().clamp(-1, 1)
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(1, 2, 3, 0) # T, H, W, C
    torchvision.io.write_video(os.path.join(save_dir, filename), video, fps=8, video_codec='h264', options={'crf': '10'})

def get_parser():
    parser = argparse.ArgumentParser(description="Unified script for final report data generation.")
    parser.add_argument("--mode", type=str, required=True, choices=['baseline', 'lora', 'student', 'ptq'])
    parser.add_argument("--output_dir", type=str, default="./final_report_data")
    parser.add_argument("--dataset_path", type=str, help="Path to ATD-12K dataset (required unless using custom_frames_dir).")
    parser.add_argument("--sample_indices", type=int, nargs='+', default=None, help="Space-separated list of sample indices to run.")
    parser.add_argument("--custom_frames_dir", type=str, default=None, help="Path to a folder with '0000.png' and '0001.png' for a custom run.")
    
    # Model Paths
    parser.add_argument("--config", type=str, default="configs/inference_512_v1.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/tooncrafter_512_interp_v1/model.ckpt")
    parser.add_argument("--lora_ckpt_dir", type=str, default="lora_rerun_final/lora_best")
    parser.add_argument("--student_config", type=str, default="configs/student_inference_512_v2.0.yaml")
    parser.add_argument("--student_ckpt", type=str, default="kd_lora_corrected/best/student.ckpt")
    parser.add_argument("--ptq_ckpt", type=str, default="final_results/quantized/quantized_model_fp16.ckpt")

    # Parameters
    parser.add_argument("--frame_stride", type=int, default=10, help="Frame stride for conditioning.")
    parser.add_argument("--lora_scale", type=float, default=0.15)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    return parser

def _apply_norm_cast_hooks(model: torch.nn.Module) -> int:
    """Keep normalization layers in parameter dtype and restore output precision."""
    model_dtype = next(model.parameters()).dtype
    norm_types = (
        torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.LayerNorm,
        torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d,
    )
    n_norm = 0
    for mod in model.modules():
        if isinstance(mod, norm_types):
            mod.float()  # keep parameters in float32 for stability
            weight_dtype = (
                mod.weight.dtype if getattr(mod, "weight", None) is not None
                else torch.float32
            )
            def _pre_hook(m, inp, dtype=weight_dtype):
                if isinstance(inp, tuple) and len(inp) > 0:
                    return (inp[0].to(dtype),) + inp[1:]
                return inp
            def _post_hook(m, inp, out, dtype=model_dtype):
                return out.to(dtype)
            mod.register_forward_pre_hook(_pre_hook)
            mod.register_forward_hook(_post_hook)
            n_norm += 1
    return n_norm

def main(args):
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Unified Model Loading Logic ---
    print(f"\n[INFO] Preparing model for mode: '{args.mode}'")
    config_path = args.student_config if args.mode == 'student' else args.config
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    checkpoint_size_gb = 0.0

    if args.mode == 'baseline':
        model_name = "Baseline (FP32)"
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"])
        checkpoint_size_gb = os.path.getsize(args.ckpt_path) / (1024**3)
    
    elif args.mode == 'lora':
        model_name = f"Teacher (LoRA @ {args.lora_scale})"
        model.load_state_dict(torch.load(args.ckpt_path, map_location="cpu")["state_dict"])
        model.model.diffusion_model = PeftModel.from_pretrained(model.model.diffusion_model, args.lora_ckpt_dir, is_trainable=False)
        l_cfg = OmegaConf.load(os.path.join(args.lora_ckpt_dir, "adapter_config.json"))
        l_alpha = l_cfg.get("lora_alpha", 16)
        for mod in model.model.diffusion_model.modules():
            if hasattr(mod, "lora_A"):
                rank = mod.r.get('default', mod.r)
                scaling_value = l_alpha / rank * args.lora_scale
                mod.scaling = {'default': scaling_value}
        # For LoRA, the "checkpoint size" is the size of the new, trained adapters
        lora_adapter_path = Path(args.lora_ckpt_dir) / "adapter_model.safetensors"
        checkpoint_size_gb = lora_adapter_path.stat().st_size / (1024**3)

    elif args.mode == 'student':
        model_name = "Student (Distilled)"
        model.load_state_dict(torch.load(args.student_ckpt, map_location="cpu")["state_dict"])
        checkpoint_size_gb = os.path.getsize(args.student_ckpt) / (1024**3)

    elif args.mode == 'ptq':
        model_name = "PTQ (FP16)"
        model = model.half()
        model.load_state_dict(torch.load(args.ptq_ckpt, map_location="cpu"))
        checkpoint_size_gb = os.path.getsize(args.ptq_ckpt) / (1024**3)
    
    model = model.to(device).eval()
    print(f"[INFO] Model '{model_name}' loaded successfully. Checkpoint Size: {checkpoint_size_gb:.3f} GB")

    if args.mode == 'ptq':
        print("[INFO] Applying FP16 compatibility hooks for PTQ mode.")
        
        # This hook function ensures that the input to a layer is cast to match the layer's weight dtype.
        def _cast_input_to_weight_dtype(mod, inp):
            if isinstance(inp, tuple) and len(inp) > 0:
                x = inp[0]
                tgt = mod.weight.dtype
                if x.dtype != tgt:
                    return (x.to(tgt),) + inp[1:]
            return None # Do nothing if input is not as expected

        n_lin, n_conv = 0, 0
        for m in model.model.diffusion_model.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                m.register_forward_pre_hook(_cast_input_to_weight_dtype)
                if isinstance(m, torch.nn.Linear): n_lin += 1
                else: n_conv += 1
        print(f"[FIX] Registered dtype-cast hooks on {n_lin} Linear and {n_conv} Conv layers in UNet.")

        # This second set of hooks ensures normalization layers remain in FP32 for stability.
        n_norm = _apply_norm_cast_hooks(model.model.diffusion_model)
        print(f"[FIX] Registered stability hooks on {n_norm} normalization layers.")
        
        # Also apply hooks to the image projection model
        if hasattr(model, "image_proj_model"):
            for m in model.image_proj_model.modules():
                 if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                    m.register_forward_pre_hook(_cast_input_to_weight_dtype)

    # --- Unified Dataset Loading Logic ---
    if args.custom_frames_dir:
        dataset = CustomFramesDataset(args.custom_frames_dir, size=(args.height, args.width))
        print(f"[INFO] Evaluating on custom frames from: {args.custom_frames_dir}")
    else:
        if not args.dataset_path:
            raise ValueError("Must provide --dataset_path if not using --custom_frames_dir")
        full_dataset = ATD12K_Dataset(args.dataset_path, video_size=(args.height, args.width), split='test')
        indices = args.sample_indices if args.sample_indices else range(5) # random.sample(range(2000), 10)
        dataset = Subset(full_dataset, indices)
        print(f"[INFO] Evaluating on {len(dataset)} specific extreme example frames: {indices}")

    dataloader = DataLoader(dataset, batch_size=1)
    
    # --- Performance Measurement ---
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    total_time, total_frames = 0, 0
    video_save_dir = os.path.join(args.output_dir, args.mode)
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), desc=f"Evaluating {model_name}", total=len(dataset)):
            start_frame = batch['start_frame'].to(device)
            end_frame = batch['end_frame'].to(device)
            videos = torch.cat([
                start_frame.unsqueeze(2).repeat(1, 1, args.video_length // 2, 1, 1),
                end_frame.unsqueeze(2).repeat(1, 1, args.video_length // 2, 1, 1)
            ], dim=2)
            prompts = batch.get("prompt", [""])

            torch.cuda.synchronize(); start_time = time.perf_counter()
            
            if args.mode == 'ptq': videos = videos.half()
            noise_shape = [1, model.model.diffusion_model.out_channels, args.video_length, args.height // 8, args.width // 8]
            if isinstance(prompts, str): prompts = [prompts]
            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, ddim_steps=50, interp=True, fs=args.frame_stride)
            
            torch.cuda.synchronize(); end_time = time.perf_counter()
            
            total_time += (end_time - start_time); total_frames += args.video_length
            
            if args.custom_frames_dir:
                sample_name = os.path.basename(args.custom_frames_dir)
                filename = f"custom_{sample_name}.mp4"
            else:
                if args.sample_indices:
                    filename = f"sample_idx_{args.sample_indices[i]}.mp4"
                else: 
                    ind = range(5)
                    filename = f"sample_idx_{ind[i]}.mp4"
            save_video(batch_samples[0, 0], filename, video_save_dir)

    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    avg_ms_per_frame = (total_time / total_frames) * 1000 if total_frames > 0 else 0
    
    log_performance_metrics(args.output_dir, model_name, checkpoint_size_gb, peak_vram_gb, avg_ms_per_frame)
    print(f"[RESULT] Checkpoint Size: {checkpoint_size_gb:.3f} GB | Peak VRAM: {peak_vram_gb:.2f} GB | Speed: {avg_ms_per_frame:.2f} ms/frame")

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)