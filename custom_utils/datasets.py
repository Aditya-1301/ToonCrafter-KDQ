import os
import glob
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class ATD12K_Dataset(Dataset):
    def __init__(self, root_dir, video_size=(320, 512), split='train'):
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        self.split = split
        self.video_size = video_size

        if self.split == 'train':
            self.image_path = os.path.join(root_dir, 'train_10k')
            self.triplet_folders = sorted(glob.glob(os.path.join(self.image_path, '*')))
        else:
            self.image_path = os.path.join(root_dir, 'test_2k_540p')
            self.annotation_path = os.path.join(root_dir, 'test_2k_annotations')
            self.triplet_folders = sorted(glob.glob(os.path.join(self.image_path, '*')))

        self.transform = transforms.Compose([
            transforms.Resize(self.video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.triplet_folders)


    def __getitem__(self, idx):
        """Return a triplet (start, middle‑GT, end) and prompt.
    
        * Robustly picks frame1/2/3 with png/jpg/jpeg extension.
        * Raises a clear error if any frame is missing.
        """
        triplet_folder_path = self.triplet_folders[idx]
        base_triplet_name = os.path.basename(triplet_folder_path)
    
        def _resolve_frame(basename: str) -> str:
            """Return the existing path for basename.[png|jpg|jpeg] or raise."""
            for ext in ("png", "jpg", "jpeg"):
                candidate = os.path.join(triplet_folder_path, f"{basename}.{ext}")
                if os.path.exists(candidate):
                    return candidate
            raise FileNotFoundError(
                f"Could not find {basename} with any supported extension in {triplet_folder_path}"
            )
    
        frame1_path = _resolve_frame("frame1")
        frame2_path = _resolve_frame("frame2")
        frame3_path = _resolve_frame("frame3")
    
        # load images and apply transforms
        start_frame_img = Image.open(frame1_path).convert("RGB")
        middle_frame_img = Image.open(frame2_path).convert("RGB")
        end_frame_img = Image.open(frame3_path).convert("RGB")
    
        start_frame = self.transform(start_frame_img)
        ground_truth_middle = self.transform(middle_frame_img)
        end_frame = self.transform(end_frame_img)
    
        # default prompt; replace with annotation for test split
        prompt = "a cartoon animation frame"
        if self.split == "test":
            annotation_folder = os.path.join(self.annotation_path, base_triplet_name)
            json_files = glob.glob(os.path.join(annotation_folder, "*.json"))
            if json_files:
                json_path = json_files[0]
                try:
                    with open(json_path, "r") as f:
                        annotation_data = json.load(f)
                    prompt = annotation_data["general_motion_type"]
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Warning: Error parsing {json_path}: {e}. Using default prompt.")
            else:
                print(f"Warning: No JSON file found in {annotation_folder}. Using default prompt.")
    
        return {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ground_truth_middle": ground_truth_middle,
            "prompt": prompt,
        }