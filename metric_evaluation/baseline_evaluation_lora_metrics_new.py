import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
import torch
import torchvision
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import lpips
from pytorch_lightning import seed_everything
from peft import PeftModel 

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from custom_utils.datasets import ATD12K_Dataset
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config

def save_video_result(video_tensor, filename, save_dir, fps=8):
    video_tensor = video_tensor.detach().cpu()
    video_tensor = torch.clamp(video_tensor.float(), -1., 1.)
    video_tensor = (video_tensor + 1.0) / 2.0
    video_tensor = (video_tensor * 255).to(torch.uint8).permute(1, 2, 3, 0) # T, H, W, C
    
    path = os.path.join(save_dir, filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torchvision.io.write_video(path, video_tensor, fps=fps, video_codec='h264', options={'crf': '10'})

def load_model_checkpoint(model, ckpt, lora_ckpt_dir=None):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()): state_dict = state_dict["state_dict"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("model.model.diffusion_model."): new_sd[k.replace("model.model.diffusion_model.", "model.diffusion_model.")] = v
        else: new_sd[k] = v
    try: model.load_state_dict(new_sd, strict=True)
    except RuntimeError as e: model.load_state_dict(new_sd, strict=False)
    print('>>> Base model checkpoint loaded.')
    if lora_ckpt_dir:
        print(f">>> Loading LoRA weights from: {lora_ckpt_dir}")
        unet = model.model.diffusion_model
        lora_model = PeftModel.from_pretrained(unet, lora_ckpt_dir)
        model.model.diffusion_model = lora_model
        print('>>> LoRA weights loaded successfully.')
    return model

# (All other helper functions from the original script are removed for clarity as they are not used in this flow)
# ... image_guided_synthesis is the key function from the original script that we need
def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)
    if not text_input: prompts = [""]*batch_size
    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        # Don't need hidden states for simple synthesis
        z = get_latent_z(model, videos)
        if loop or interp:
            img_cat_cond = torch.zeros_like(z); img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]; img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
        else:
            img_cat_cond = z[:,:,:1,:,:]; img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond]
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]; uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)); uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid': uc["c_concat"] = [img_cat_cond]
    else: uc = None
    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(S=ddim_steps, conditioning=cond, batch_size=batch_size, shape=noise_shape[1:], verbose=False, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc, eta=ddim_eta, cfg_img=cfg_img, fs=fs, timestep_spacing=timestep_spacing, guidance_rescale=guidance_rescale, **kwargs)
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def run_evaluation(args):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda()
    model = load_model_checkpoint(model, args.ckpt_path, args.lora_ckpt_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    full_test_dataset = ATD12K_Dataset(root_dir=args.dataset_path, video_size=(args.height, args.width), split='test')
    
    # We force a small subset for diagnostic purposes
    if not args.full_eval:
        print("\n*** RUNNING IN DIAGNOSTIC MODE (5 SAMPLES) TO VISUALIZE OUTPUT ***\n")
        test_dataset = torch.utils.data.Subset(full_test_dataset, list(range(5)))
    else:
        print("\n*** RUNNING FULL EVALUATION (2000 SAMPLES) ***\n")
        test_dataset = full_test_dataset

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    sample_save_dir = os.path.join(args.output_dir, "evaluation_videos")
    os.makedirs(sample_save_dir, exist_ok=True)
    all_psnr, all_ssim, all_lpips = [], [], []
    noise_shape = [1, model.model.diffusion_model.out_channels, args.video_length, args.height // 8, args.width // 8]
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in tqdm(enumerate(dataloader), desc="Evaluating Model", total=len(test_dataset)):
            start_frame = batch['start_frame'].unsqueeze(2).repeat(1, 1, args.video_length//2, 1, 1)
            end_frame = batch['end_frame'].unsqueeze(2).repeat(1, 1, args.video_length//2, 1, 1)
            videos = torch.cat([start_frame, end_frame], dim=2).cuda()
            prompts = [batch['prompt'][0]]
            batch_samples_6d = image_guided_synthesis(model, prompts, videos, noise_shape, 1, args.ddim_steps, args.ddim_eta, args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)
            
            # THE FIX: Select the first sample from the n_samples dimension
            generated_video_5d = batch_samples_6d[0, 0] # Shape: [C, T, H, W]
            
            if not args.full_eval:
                filename = f"diagnostic_sample_idx_{i}.mp4"
                save_video_result(generated_video_5d, filename, sample_save_dir, fps=8)
            
            middle_idx = args.video_length // 2
            generated_frame_tensor = generated_video_5d[:, middle_idx, :, :]
            ground_truth_tensor = batch['ground_truth_middle'][0]
            gen_np = ((generated_frame_tensor.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            gt_np = ((ground_truth_tensor.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            all_psnr.append(psnr(gt_np, gen_np, data_range=255))
            all_ssim.append(ssim(cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY), cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY), data_range=255))
            lpips_score = lpips_loss_fn(generated_frame_tensor.unsqueeze(0).to(device), ground_truth_tensor.unsqueeze(0).to(device))
            all_lpips.append(lpips_score.item())

    avg_psnr = np.mean(all_psnr); avg_ssim = np.mean(all_ssim); avg_lpips = np.mean(all_lpips)
    print("\n" + "="*40 + "\n      EVALUATION COMPLETE\n" + "="*40)
    print(f"  PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    print("="*40 + "\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_eval", action="store_true", help="Run full evaluation on all 2000 test samples.")
    parser.add_argument("--lora_ckpt_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
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
    parser.add_argument("--perframe_ae", action='store_true', default=False)
    parser.add_argument("--loop", action='store_true', default=False)
    parser.add_argument("--interp", action='store_true', default=True)
    return parser

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"@LoRA Model Evaluation: {now}")
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_evaluation(args)