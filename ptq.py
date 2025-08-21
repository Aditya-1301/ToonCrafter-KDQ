import argparse, os, sys, torch, gc
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.utils import instantiate_from_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_ckpt_path", type=str, required=True)
    parser.add_argument("--quantization_level", type=str, default="fp16", choices=['fp16', 'bf16'])
    args = parser.parse_args()

    # Load the original FP32 model
    print(f"Loading base model from {args.ckpt_path}...")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config.model).cpu()
    state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    del state_dict; gc.collect()  # Be extra clean with memory
    print("    ✔ Base model loaded.")

    # Apply conversion/quantization
    print(f"Applying {args.quantization_level.upper()} conversion...")
    unet = model.model.diffusion_model
    if args.quantization_level == 'fp16':
        quantized_unet = unet.half()
    elif args.quantization_level == 'bf16':
        quantized_unet = unet.to(torch.bfloat16)
    else:
        raise ValueError("Unsupported quantization level")
    model.model.diffusion_model = quantized_unet
    print("    ✔ Conversion complete.")

    # Save the new checkpoint
    os.makedirs(os.path.dirname(args.output_ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_ckpt_path)
    size_mb = os.path.getsize(args.output_ckpt_path) / (1024 * 1024)
    print(f"    ✔ Converted model saved to: {args.output_ckpt_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()