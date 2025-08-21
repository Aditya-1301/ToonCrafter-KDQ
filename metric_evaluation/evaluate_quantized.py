import argparse, os, sys, torch, gc
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import lpips
import torchvision
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset
from scripts.evaluation.inference import image_guided_synthesis


def _apply_norm_cast_hooks(model: torch.nn.Module) -> int:
    """Keep normalization layers in parameter dtype and restore output precision.

    This registers forward pre/post hooks on every normalization layer so that
    incoming activations are cast to the layer's ``weight.dtype`` (typically
    ``float32``) before computation and outputs are cast back to the model's
    overall dtype (e.g. ``float16``).
    """
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
                return (inp[0].to(dtype),)

            def _post_hook(m, inp, out, dtype=model_dtype):
                return out.to(dtype)

            mod.register_forward_pre_hook(_pre_hook)
            mod.register_forward_hook(_post_hook)
            n_norm += 1
    return n_norm


def save_results_separate(prompt, samples, filename, outdir, fps=10, loop=False):
    """Save generated videos to ``outdir`` as MP4 files.

    Parameters
    ----------
    prompt: list or str
        Prompt corresponding to the samples (unused but kept for parity).
    samples: torch.Tensor
        Tensor with shape ``[C, T, H, W]`` or ``[1, C, T, H, W]`` in ``[-1, 1]`` range.
    filename: str
        Filename for the saved video.
    outdir: str
        Directory where the video will be written.
    fps: int
        Frames per second for the output video.
    loop: bool
        Whether to drop the last frame to make a looping video.
    """
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    os.makedirs(outdir, exist_ok=True)
    video = samples.detach().cpu()
    if video.ndim == 5:  # [B, C, T, H, W] -> assume B==1
        video = video[0]
    if loop:
        video = video[:, :-1, ...]
    video = torch.clamp(video.float(), -1., 1.)
    grid = (video + 1.0) / 2.0
    grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()  # THWC
    path = os.path.join(outdir, filename)
    torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})


def run_evaluation(args, model, device):
    """Runs the standard evaluation loop on the provided model."""
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    psnr_m = PeakSignalNoiseRatio(data_range=1.).to(device)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.).to(device)

    full_dataset = ATD12K_Dataset(args.dataset_path, video_size=(args.height, args.width), split='test')

    if args.debug:
        print("\n[INFO] Running in DEBUG mode on a small subset of 5 samples.")
        dataset = Subset(full_dataset, list(range(5)))
    elif args.num_samples is not None:
        # If --num_samples is provided, use that many samples.
        # We select them from the start of the dataset for consistency.
        num_to_run = min(args.num_samples, len(full_dataset))
        print(f"\n[INFO] Running evaluation on a REPRESENTATIVE SUBSET of {num_to_run} samples.")
        dataset = Subset(full_dataset, list(range(num_to_run)))
    else:
        # Default to the full evaluation
        print(f"\n[INFO] Running FULL evaluation on {len(full_dataset)} samples.")
        dataset = full_dataset

    dataloader = DataLoader(dataset, batch_size=args.bs, num_workers=4)

    all_psnr, all_ssim, all_lpips = [], [], []

    sample_dir = None
    if args.output_dir and args.save_every_n > 0:
        sample_dir = os.path.join(args.output_dir, "evaluation_samples")
        os.makedirs(sample_dir, exist_ok=True)
        print(f"Sample videos will be saved in: {sample_dir}")

    sample_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Quantized Model"):
            # current batch size may be smaller than args.bs for the last iteration
            bsz = batch['start_frame'].shape[0]

            videos = torch.cat([
                batch['start_frame'].unsqueeze(2).repeat(1, 1, args.video_length // 2, 1, 1),
                batch['end_frame'].unsqueeze(2).repeat(1, 1, args.video_length // 2, 1, 1)
            ], dim=2).to(device)
            prompts = batch.get("prompt", [])
            if isinstance(prompts, (str, bytes)):
                prompts = [prompts]
            else:
                prompts = list(prompts)
            if len(prompts) < bsz:
                prompts.extend([""] * (bsz - len(prompts)))
            elif len(prompts) > bsz:
                prompts = prompts[:bsz]
            gt_middle_frame = batch['ground_truth_middle'].to(device)

            # --- Cast input tensor to model's datatype ---
            videos = videos.to(model.dtype)

            noise_shape = [
                bsz,
                model.model.diffusion_model.out_channels,
                args.video_length,
                args.height // 8,
                args.width // 8,
            ]

            batch_samples = image_guided_synthesis(
                model, prompts, videos, noise_shape,
                n_samples=1, ddim_steps=args.ddim_steps, ddim_eta=args.ddim_eta,
                unconditional_guidance_scale=args.unconditional_guidance_scale, cfg_img=args.cfg_img,
                fs=args.frame_stride, text_input=args.text_input, multiple_cond_cfg=args.multiple_cond_cfg,
                loop=args.loop, interp=args.interp, timestep_spacing=args.timestep_spacing,
                guidance_rescale=args.guidance_rescale
            )

            gen_video = batch_samples[:, 0]
            middle_idx = args.video_length // 2 - 1
            gen_middle_frame = gen_video[:, :, middle_idx, :, :]

            # --- Ensure metrics are calculated in float32 ---
            p01, m01 = (gen_middle_frame.float() + 1) / 2, (gt_middle_frame.float() + 1) / 2
            all_psnr.append(psnr_m(p01, m01).item())
            all_ssim.append(ssim_m(p01, m01).item())
            all_lpips.append(lpips_fn(gen_middle_frame.float(), gt_middle_frame.float()).mean().item())

            # Save a sample video from the current batch if the condition is met.
            # We use sample_idx to track the overall sample count.
            if sample_dir and (sample_idx // args.bs) % (args.save_every_n // args.bs) == 0:
                # Save the first sample from the current batch
                video_to_save = gen_video[0:1] # Shape: [1, C, T, H, W]
                prompt_to_save = [prompts[0]]
                filename = f"sample_{sample_idx}.mp4"
                
                print(f"\n[INFO] Saving sample video: {filename}")
                save_results_separate(
                    prompt_to_save,
                    video_to_save,
                    filename,
                    sample_dir,
                    fps=args.fps,
                    loop=args.loop,
                )
            
            sample_idx += bsz

    avg_psnr = np.mean(all_psnr); avg_ssim = np.mean(all_ssim); avg_lpips = np.mean(all_lpips)
    print("\n" + "="*40 + f"\n      {args.quantization_level.upper()} EVALUATION COMPLETE\n" + "="*40)
    print(f"  PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    print("="*40 + "\n")


def main():
    parser = argparse.ArgumentParser()
    # --- Complete Argument Parser ---
    parser.add_argument("--quantized_ckpt_path", type=str, required=True)
    parser.add_argument("--quantization_level", type=str, default="fp16", choices=['fp16', 'bf16'])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--debug", action='store_true', help="Run on a small subset for debugging.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save generated videos.")
    parser.add_argument("--save_every_n", type=int, default=0, help="Save one batch every n iterations (0 disables).")
    parser.add_argument("--fps", type=int, default=8, help="FPS for saved videos.")

    # Arguments required by the original helper functions
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)
    parser.add_argument("--frame_stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--text_input", action='store_true', default=True)
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False)
    parser.add_argument("--cfg_img", type=float, default=None)
    parser.add_argument("--timestep_spacing", type=str, default="uniform")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--loop", action='store_true', default=False)
    parser.add_argument("--interp", action='store_true', default=True)
    parser.add_argument("--num_samples", type=int, default=None, help="Run evaluation on a specific number of samples from the test set.")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- (Model loading logic is correct and does not need changes) ---
    print(f"Loading {args.quantization_level.upper()} model from {args.quantized_ckpt_path}")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).cpu()

    if args.quantization_level == 'fp16':
        model = model.half()
    elif args.quantization_level == 'bf16':
        model = model.to(torch.bfloat16)

    model.load_state_dict(torch.load(args.quantized_ckpt_path, map_location="cpu"))
    print("    ✔ Converted model loaded successfully.")

    model = model.to(device)

    if args.quantization_level == 'fp16':
        print("[FIX] Manually ensuring time_embed module is FP16...")
        model.model.diffusion_model.time_embed = model.model.diffusion_model.time_embed.half()
        print("    ✔ time_embed module successfully cast.")
    elif args.quantization_level == 'bf16':
        print("[FIX] Manually ensuring time_embed module is BF16...")
        model.model.diffusion_model.time_embed = model.model.diffusion_model.time_embed.to(torch.bfloat16)
        print("    ✔ time_embed module successfully cast.")

    def _cast_input_to_weight_dtype(mod, inp):
        x = inp[0]
        tgt = mod.weight.dtype
        if x.dtype != tgt:
            return (x.to(tgt),)
        return None

    n_lin, n_conv = 0, 0
    for _, m in model.model.diffusion_model.named_modules():
        if isinstance(m, torch.nn.Linear):
            m.register_forward_pre_hook(_cast_input_to_weight_dtype)
            n_lin += 1
        elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            m.register_forward_pre_hook(_cast_input_to_weight_dtype)
            n_conv += 1
    print(f"[FIX] Registered dtype-cast hooks on {n_lin} Linear and {n_conv} Conv layers in UNet.")

    n_norm = _apply_norm_cast_hooks(model.model.diffusion_model)
    print(f"[FIX] Registered dtype-cast hooks on {n_norm} normalization layers.")

    try:
        te_w = model.model.diffusion_model.time_embed[0].weight.dtype
        print(f"[DBG] time_embed Linear weight dtype: {te_w}")
    except Exception:
        pass
    try:
        fe_w = model.model.diffusion_model.fps_embedding[0].weight.dtype
        print(f"[DBG] fps_embedding Linear weight dtype: {fe_w}")
    except Exception:
        pass

    if hasattr(model, "image_proj_model") and isinstance(model.image_proj_model, torch.nn.Module):
        n_lin = n_conv = 0
        for m in model.image_proj_model.modules():
            if isinstance(m, torch.nn.Linear):
                m.register_forward_pre_hook(_cast_input_to_weight_dtype); n_lin += 1
            elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                m.register_forward_pre_hook(_cast_input_to_weight_dtype); n_conv += 1
        print(f"[FIX] image_proj_model hooks: {n_lin} Linear, {n_conv} Conv (inputs->weight dtype)")

    model.eval()

    run_evaluation(args, model, device)

if __name__ == "__main__":
    main()