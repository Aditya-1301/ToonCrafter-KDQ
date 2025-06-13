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

# --- MODIFICATION 1: ADD REQUIRED IMPORTS ---
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
import lpips
# This now imports from the library, as it should.
from pytorch_lightning import seed_everything
from custom_utils.datasets import ATD12K_Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config

# --- ALL HELPER FUNCTIONS BELOW ARE UNTOUCHED FROM YOUR WORKING SCRIPT ---
def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items(): new_pl_sd[k] = v
            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding"); new_pl_sd[new_key] = new_pl_sd[k]; del new_pl_sd[k]
            model.load_state_dict(new_pl_sd, strict=True)
    else:
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys(): new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model

def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z

def get_latent_z_with_hidden_states(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    encoder_posterior, hidden_states = model.first_stage_model.encode(x, return_hidden_states=True)
    hidden_states_first_last = []
    for hid in hidden_states:
        hid = rearrange(hid, '(b t) c h w -> b c t h w', t=t)
        hid_new = torch.cat([hid[:, :, 0:1], hid[:, :, -1:]], dim=2)
        hidden_states_first_last.append(hid_new)
    z = model.get_first_stage_encoding(encoder_posterior).detach()
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z, hidden_states_first_last

def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)
    if not text_input:
        prompts = [""]*batch_size
    img = videos[:,:,0] #bchw
    img_emb = model.embedder(img)
    img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z, hs = get_latent_z_with_hidden_states(model, videos)
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
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None
    additional_decode_kwargs = {'ref_context': hs}
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})
    batch_variants = []
    for _ in range(n_samples):
        samples, _ = ddim_sampler.sample(S=ddim_steps, conditioning=cond, batch_size=batch_size, shape=noise_shape[1:], verbose=False, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=uc, eta=ddim_eta, cfg_img=cfg_img, fs=fs, timestep_spacing=timestep_spacing, guidance_rescale=guidance_rescale, **kwargs)
        batch_images = model.decode_first_stage(samples, timesteps=samples.shape[2], **additional_decode_kwargs)
        index = list(range(samples.shape[2])); del index[1]; del index[-2]
        samples = samples[:,:,index,:,:]
        batch_images_middle = model.decode_first_stage(samples, timesteps=samples.shape[2], **additional_decode_kwargs)
        batch_images[:,:,batch_images.shape[2]//2-1:batch_images.shape[2]//2+1] = batch_images_middle[:,:,batch_images.shape[2]//2-2:batch_images.shape[2]//2]
        batch_variants.append(batch_images)
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def save_results_seperate(prompt, samples, filename, fakedir, fps=10, loop=False):
    prompt = prompt[0] if isinstance(prompt, list) else prompt
    videos = [samples]
    savedirs = [fakedir]
    for idx, video in enumerate(videos):
        if video is None:
            continue
        video = video.detach().cpu()
        if loop:
            video = video[:,:,:-1,...]
        video = torch.clamp(video.float(), -1., 1.)
        n = video.shape[0]
        for i in range(n):
            grid = video[i,...]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(1, 2, 3, 0) #thwc
            path = os.path.join(savedirs[idx].replace('samples', 'samples_separate'), f'{filename.split(".")[0]}_sample{i}.mp4')
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))
            torchvision.io.write_video(path, grid, fps=fps, video_codec='h264', options={'crf': '10'})

# --- MODIFICATION 2: This is the new evaluation driver function ---
def run_evaluation(args):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda()
    model.perframe_ae = args.perframe_ae
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    full_test_dataset = ATD12K_Dataset(root_dir=args.dataset_path, video_size=(args.height, args.width), split='test')
    if args.debug:
        print("\n*** DEBUG MODE: USING 5 SAMPLES ***\n")
        test_dataset = torch.utils.data.Subset(full_test_dataset, list(range(5)))
    else:
        test_dataset = full_test_dataset
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(test_dataset)} samples.")
    
    sample_save_dir = os.path.join(args.output_dir, "evaluation_samples")
    os.makedirs(sample_save_dir, exist_ok=True)
    print(f"Sample videos will be saved in: {sample_save_dir}")

    all_psnr, all_ssim, all_lpips = [], [], []
    worst_lpips_score = -1
    worst_sample_data = None

    noise_shape = [1, model.model.diffusion_model.out_channels, args.video_length, args.height // 8, args.width // 8]
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for i, batch in tqdm(enumerate(dataloader), desc="Evaluating Baseline"):
            try:
                # The 'repeat' logic here was wrong. It should use unsqueeze.
                start_frame = batch['start_frame'].unsqueeze(2).repeat(1, 1, args.video_length//2, 1, 1)
                end_frame = batch['end_frame'].unsqueeze(2).repeat(1, 1, args.video_length//2, 1, 1)
                videos = torch.cat([start_frame, end_frame], dim=2).cuda()
                prompts = [batch['prompt'][0]]
                
                batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, 1, args.ddim_steps, args.ddim_eta,
                                                     args.unconditional_guidance_scale, args.cfg_img, args.frame_stride, 
                                                     args.text_input, args.multiple_cond_cfg, args.loop, args.interp, 
                                                     args.timestep_spacing, args.guidance_rescale)
                
                generated_video = batch_samples[0, 0]
                middle_idx = args.video_length // 2
                generated_frame_tensor = generated_video[:, middle_idx, :, :]
                ground_truth_tensor = batch['ground_truth_middle'][0]

                gen_np = ((generated_frame_tensor.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                gt_np = ((ground_truth_tensor.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                all_psnr.append(psnr(gt_np, gen_np, data_range=255))
                all_ssim.append(ssim(cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY), cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY), data_range=255))
                
                lpips_score = lpips_loss_fn(generated_frame_tensor.unsqueeze(0).to(device), ground_truth_tensor.unsqueeze(0).to(device))
                current_lpips = lpips_score.item()
                all_lpips.append(current_lpips)

                if args.debug:
                    if worst_lpips_score == -1 or current_lpips > worst_lpips_score:
                        worst_lpips_score = current_lpips
                        worst_sample_data = {"samples": batch_samples, "prompt": prompts, "filename": f"worst_debug_sample_idx_{i}.mp4"}
                elif i % args.save_every_n == 0:
                    save_results_seperate(prompts, batch_samples[0], f"sample_{i}.mp4", sample_save_dir, fps=8)
                    print(f"\nSaved sample video for index {i}")

            except Exception as e:
                print(f"--- FAILED ON A SAMPLE: {e} ---"); import traceback; traceback.print_exc()

    if args.debug and worst_sample_data is not None:
        print(f"\nSaving worst debug sample: {worst_sample_data['filename']} (LPIPS: {worst_lpips_score:.4f})")
        save_results_seperate(worst_sample_data['prompt'], worst_sample_data['samples'][0], worst_sample_data['filename'], sample_save_dir, fps=8)

    avg_psnr = np.mean(all_psnr) if all_psnr else 0
    avg_ssim = np.mean(all_ssim) if all_ssim else 0
    avg_lpips = np.mean(all_lpips) if all_lpips else 0
    print("\n" + "="*40 + "\n      BASELINE EVALUATION COMPLETE\n" + "="*40)
    print(f"  Successfully evaluated: {len(all_psnr)} / {len(test_dataset)}")
    print(f"  Average PSNR: {avg_psnr:.2f} (Higher is better)")
    print(f"  Average SSIM: {avg_ssim:.4f} (Higher is better)")
    print(f"  Average LPIPS: {avg_lpips:.4f} (Lower is better)")
    print("="*40 + "\n")

# --- MODIFICATION 3: This is the new, minimal parser ---
def get_parser():
    parser = argparse.ArgumentParser()
    # Arguments for our new script
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save sample videos.")
    parser.add_argument("--save_every_n", type=int, default=100, help="Save a sample video every N iterations.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    # Arguments required by the original helper functions
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

# --- MODIFICATION 4: The main execution block ---
if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"@Baseline Evaluation: {now}")
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_evaluation(args)