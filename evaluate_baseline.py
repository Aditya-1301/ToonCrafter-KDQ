import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import cv2
from custom_utils.datasets import ATD12K_Dataset

sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config


def get_filelist(data_dir, postfixes):
    patterns = [os.path.join(data_dir, f"*.{postfix}") for postfix in postfixes]
    file_list = []
    for pattern in patterns:
        file_list.extend(glob.glob(pattern))
    file_list.sort()
    return file_list

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
        
        # The VAE Decoder for video needs to know the number of frames `t`.
        batch_images = model.decode_first_stage(samples, timesteps=samples.shape[2], **additional_decode_kwargs)
        
        index = list(range(samples.shape[2])); del index[1]; del index[-2]
        samples = samples[:,:,index,:,:]
        
        batch_images_middle = model.decode_first_stage(samples, timesteps=samples.shape[2], **additional_decode_kwargs)
        
        batch_images[:,:,batch_images.shape[2]//2-1:batch_images.shape[2]//2+1] = batch_images_middle[:,:,batch_images.shape[2]//2-2:batch_images.shape[2]//2]
        batch_variants.append(batch_images)
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def run_evaluation(args):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda()
    model.perframe_ae = args.perframe_ae
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    
    full_test_dataset = ATD12K_Dataset(root_dir=args.dataset_path, video_size=(args.height, args.width), split='test')
    if args.debug:
        print("\n*** DEBUG MODE: USING 5 SAMPLES ***\n")
        test_dataset = torch.utils.data.Subset(full_test_dataset, list(range(5)))
    else:
        test_dataset = full_test_dataset
    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    print(f"Evaluating on {len(test_dataset)} samples.")
    
    all_psnr, all_ssim = [], []
    noise_shape = [1, model.model.diffusion_model.out_channels, args.video_length, args.height // 8, args.width // 8]
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in tqdm(dataloader, desc="Evaluating Baseline"):
            try:
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
                generated_frame = generated_video[:, middle_idx, :, :]
                ground_truth_frame = batch['ground_truth_middle'][0]

                gen_np = ((generated_frame.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                gt_np = ((ground_truth_frame.clamp(-1, 1) + 1.0) / 2.0 * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

                all_psnr.append(psnr(gt_np, gen_np, data_range=255))
                all_ssim.append(ssim(cv2.cvtColor(gt_np, cv2.COLOR_RGB2GRAY), cv2.cvtColor(gen_np, cv2.COLOR_RGB2GRAY), data_range=255))

            except Exception as e:
                print(f"--- FAILED ON A SAMPLE: {e} ---"); import traceback; traceback.print_exc()

    avg_psnr = np.mean(all_psnr) if all_psnr else 0; avg_ssim = np.mean(all_ssim) if all_ssim else 0
    print("\n" + "="*40 + "\n      BASELINE EVALUATION COMPLETE\n" + "="*40)
    print(f"  Successfully evaluated: {len(all_psnr)} / {len(test_dataset)}"); print(f"  Average PSNR: {avg_psnr:.2f}"); print(f"  Average SSIM: {avg_ssim:.4f}"); print("="*40 + "\n")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Root path to the ATD-12k dataset.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode on a small subset of data.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--height", type=int, default=320); parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--video_length", type=int, default=16); parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0); parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)
    parser.add_argument("--frame_stride", type=int, default=10); parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--text_input", action='store_true', default=True); parser.add_argument("--multiple_cond_cfg", action='store_true', default=False)
    parser.add_argument("--cfg_img", type=float, default=None); parser.add_argument("--timestep_spacing", type=str, default="uniform")
    parser.add_argument("--guidance_rescale", type=float, default=0.0); parser.add_argument("--perframe_ae", action='store_true', default=False)
    parser.add_argument("--loop", action='store_true', default=False); parser.add_argument("--interp", action='store_true', default=True)
    return parser

if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"@Baseline Evaluation: {now}")
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_evaluation(args)