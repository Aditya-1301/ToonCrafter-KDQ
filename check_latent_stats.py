import torch, json, random
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from custom_utils.datasets import ATD12K_Dataset
from tqdm import tqdm

# --- edit these three paths -------------------------------
CFG = "configs/inference_512_v1.0.yaml"
CKPT = "checkpoints/tooncrafter_512_interp_v1/model.ckpt"
DS   = "/home/jovyan/thesis/tooncrafter/atd12k_dataset"
student_cfg = "configs/student_inference_512_v1.0.yaml"
# -----------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = instantiate_from_config(OmegaConf.load(CFG).model).to(device)
model.load_state_dict(torch.load(CKPT, map_location="cpu")["state_dict"], strict=False)
model.eval(); model.first_stage_model.eval()

dl = torch.utils.data.DataLoader(
        ATD12K_Dataset(DS, split="train", video_size=(320,512)),
        batch_size=4, shuffle=True, num_workers=2)

m, s, n = 0., 0., 0
with torch.no_grad():
    for i, batch in enumerate(tqdm(dl, total=20)):   # 20×4 = 80 images is enough
        imgs = batch["start_frame"].to(device)        # any frame works
        z = model.encode_first_stage(imgs) / model.scale_factor
        m += z.mean().item();  s += z.std().item();  n += 1
        if i == 19: break

print({"mean": m/n, "std": s/n, "scale_factor": float(model.scale_factor)})

student = instantiate_from_config(OmegaConf.load(student_cfg).model)
print([k for k in student.state_dict().keys() if "first_stage_model" in k][:10], "...", len([k for k in student.state_dict() if "first_stage_model" in k]), "params")

