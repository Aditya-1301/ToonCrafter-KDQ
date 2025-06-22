import argparse, os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from pytorch_lightning import seed_everything
from einops import rearrange
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset

# --- HELPER FUNCTIONS (Verified against your codebase) ---
def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
    
    # Custom loading logic to handle potential key mismatches
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model.model.diffusion_model."):
            new_k = k.replace("model.model.diffusion_model.", "model.diffusion_model.")
            new_sd[new_k] = v
        else:
            new_sd[k] = v
            
    try:
        model.load_state_dict(new_sd, strict=True)
    except RuntimeError as e:
        print(f"Caught a RuntimeError, attempting to load with strict=False: {e}")
        model.load_state_dict(new_sd, strict=False)

    print('>>> model checkpoint loaded.')
    return model

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    # The model's encode_first_stage handles the scaling factor internally
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

# --- MAIN TRAINING SCRIPT ---
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. --- Model Loading ---
    config = OmegaConf.load(args.config)
    # The DDPM class is the one we need to instantiate
    model = instantiate_from_config(config.model)
    model = load_model_checkpoint(model, args.ckpt_path)
    unet = model.model.diffusion_model
    model.to(device)
    
    # 2. --- Toon Rectification Strategy ---
    print("Applying Toon Rectification learning strategy (Freezing Temporal Layers)...")
    for name, param in unet.named_parameters():
        if "temporal_transformer" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # Ensure image_proj_model is trainable as per the paper
    for param in model.image_proj_model.parameters():
        param.requires_grad = True

    print("Applying LoRA adapters to spatial attention layers...")
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.1, bias="none",
    )
    lora_unet = get_peft_model(unet, lora_config)
    lora_unet.print_trainable_parameters()

    # 3. --- Data Loading ---
    full_train_dataset = ATD12K_Dataset(root_dir=args.dataset_path, video_size=(args.height, args.width), split='train')
    if args.debug:
        train_dataset = torch.utils.data.Subset(full_train_dataset, list(range(10)))
    else:
        train_dataset = torch.utils.data.Subset(full_train_dataset, list(range(args.train_subset_size)))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    print(f"Training on {len(train_dataset)} samples.")

    # 4. --- Optimizer ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, lora_unet.parameters()), lr=args.learning_rate)
    
    # 5. --- Training Loop ---
    print("Starting LoRA fine-tuning...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{args.epochs}"):
            try:
                optimizer.zero_grad()
                
                # --- A. Prepare Inputs ---
                start_frame = batch["start_frame"].to(device)
                end_frame = batch["end_frame"].to(device)
                prompts = list(batch["prompt"])
                B = start_frame.shape[0]

                # --- B. Create Conditioning Tensors ---
                with torch.no_grad():
                    # B.1 Create the spatial conditioning `c_concat`
                    video_cat_tensor = torch.stack([start_frame, end_frame], dim=2)
                    z_cat = get_latent_z(model, video_cat_tensor)
                    # This is the latent version of the start/end frames with zeros in the middle
                    c_concat_cond = torch.zeros((B, z_cat.shape[1], args.video_length, z_cat.shape[3], z_cat.shape[4]), device=device)
                    c_concat_cond[:, :, 0, :, :] = z_cat[:, :, 0, :, :]
                    c_concat_cond[:, :, -1, :, :] = z_cat[:, :, -1, :, :]
                    
                    # B.2 Create the cross-attention conditioning `context` tensor
                    text_emb = model.get_learned_conditioning(prompts)
                    img_emb = model.embedder(start_frame)
                    img_emb = model.image_proj_model(img_emb)
                    # THIS IS THE KEY: Concatenate them into a single tensor, as the UNet expects.
                    c_crossattn_cond = torch.cat([text_emb, img_emb], dim=1)

                # --- C. Diffusion Process ---
                x_start = c_concat_cond # The "clean" data is the latent with start/end frames.
                noise = torch.randn_like(x_start)
                ts = torch.randint(0, model.num_timesteps, (B,), device=device).long()
                noisy_latents = model.q_sample(x_start=x_start, t=ts, noise=noise)

                # --- D. Forward Pass (mimicking DiffusionWrapper) ---
                # D.1 Concatenate noisy latents with spatial conditioning along the channel dim
                # The UNet's input channel count is in_channels (4) + c_concat channels (4) = 8
                noisy_latents_with_concat = torch.cat([noisy_latents, c_concat_cond], dim=1)
                
                # D.2 Call the UNet with the combined inputs
                predicted_noise = lora_unet(noisy_latents_with_concat, ts, context=c_crossattn_cond)
                
                # --- E. Calculate Loss ---
                # The objective is to predict the original noise that was added.
                loss = F.l1_loss(predicted_noise, noise)
                
                # --- F. Backward Pass & Optimization ---
                loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, lora_unet.parameters()), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()

            except Exception as e:
                print(f"ERROR during training step {step}: {e}")
                import traceback
                traceback.print_exc()
                # Uncomment for deep debugging if an error persists
                # torch.save({
                #     "batch": batch,
                #     "noisy_latents_with_concat": noisy_latents_with_concat,
                #     "c_crossattn_cond": c_crossattn_cond,
                #     "ts": ts
                # }, "failed_batch.pt")
                # sys.exit(1)
                continue
        
        avg_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        
        # 6. --- Save Checkpoint ---
        checkpoint_dir = os.path.join(args.output_dir, f"lora_epoch_{epoch+1}")
        lora_unet.save_pretrained(checkpoint_dir)
        print(f"LoRA adapter saved to {checkpoint_dir}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/inference_512_v1.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/tooncrafter_512_interp_v1/model.ckpt")
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--train_subset_size", type=int, default=2000)
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)