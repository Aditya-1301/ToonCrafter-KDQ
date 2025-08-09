import os
import glob
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd # Make sure to import pandas

class ATD12K_Dataset(Dataset):
    # ADDED: New optional arguments for returning filenames and loading CSV annotations.
    def __init__(self, root_dir, video_size=(320, 512), split='train', 
                 return_filename=False, annotations_csv=None):
        
        assert split in ['train', 'test'], "Split must be 'train' or 'test'"
        self.split = split
        self.video_size = video_size
        self.return_filename = return_filename # Store the flag

        if self.split == 'train':
            self.image_path = os.path.join(root_dir, 'train_10k')
            self.triplet_folders = sorted(glob.glob(os.path.join(self.image_path, '*')))
        else: # test split
            self.image_path = os.path.join(root_dir, 'test_2k_540p')
            self.annotation_path = os.path.join(root_dir, 'test_2k_annotations')
            self.triplet_folders = sorted(glob.glob(os.path.join(self.image_path, '*')))

        self.transform = transforms.Compose([
            transforms.Resize(self.video_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        # ADDED: Logic to load annotations from a CSV file.
        self.annotations = {}
        if annotations_csv and os.path.exists(annotations_csv):
            print(f">>> Loading annotations from {annotations_csv}")
            df = pd.read_csv(annotations_csv)
            # Create a fast lookup dictionary from filename to caption
            self.annotations = pd.Series(df.caption.values, index=df.filename).to_dict()
            print(f">>> Loaded {len(self.annotations)} captions.")

    def __len__(self):
        return len(self.triplet_folders)

    def __getitem__(self, idx):
        triplet_folder_path = self.triplet_folders[idx]
        base_triplet_name = os.path.basename(triplet_folder_path)
    
        def _resolve_frame(basename: str) -> str:
            for ext in ("png", "jpg", "jpeg"):
                candidate = os.path.join(triplet_folder_path, f"{basename}.{ext}")
                if os.path.exists(candidate): return candidate
            raise FileNotFoundError(f"Could not find {basename} in {triplet_folder_path}")
    
        frame1_path = _resolve_frame("frame1")
        frame2_path = _resolve_frame("frame2")
        frame3_path = _resolve_frame("frame3")
    
        start_frame_img = Image.open(frame1_path).convert("RGB")
        middle_frame_img = Image.open(frame2_path).convert("RGB")
        end_frame_img = Image.open(frame3_path).convert("RGB")
    
        start_frame = self.transform(start_frame_img)
        ground_truth_middle = self.transform(middle_frame_img)
        end_frame = self.transform(end_frame_img)
        
        # --- Prompt Logic ---
        # Prioritize CSV annotations if they exist for this file.
        # Fallback to JSON for the test set, then a default prompt.
        prompt = self.annotations.get(base_triplet_name)
        if prompt is None:
            if self.split == "test":
                annotation_folder = os.path.join(self.annotation_path, base_triplet_name)
                json_files = glob.glob(os.path.join(annotation_folder, "*.json"))
                if json_files:
                    try:
                        with open(json_files[0], "r") as f:
                            annotation_data = json.load(f)
                        prompt = annotation_data["general_motion_type"]
                    except (KeyError, json.JSONDecodeError):
                        prompt = "a cartoon animation frame" # Fallback
                else:
                    prompt = "a cartoon animation frame" # Fallback
            else: # train split without CSV annotation
                prompt = "a cartoon animation frame"
        
        batch = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "ground_truth_middle": ground_truth_middle,
            "prompt": prompt,
        }

        # ADDED: Conditionally add the filename to the batch if requested.
        if self.return_filename:
            batch['file_name'] = base_triplet_name
            
        return batch