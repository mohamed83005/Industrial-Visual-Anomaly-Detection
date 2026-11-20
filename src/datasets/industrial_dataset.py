"""
Industrial Dataset Loader & Splitter
Pharos University in Alexandria (AI403) - Deep Learning
"""

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class IndustrialDataset(Dataset):
    """
    Dataset loader for multi-sector industrial images (PCB, Nut, Weld, Tile).
    Supports one-class training (normal only) and evaluation (normal + defective).
    """
    def __init__(self, root_dir: str, sector: str = "01_electronics_pcb", split: str = "train", transform=None):
        self.root_dir = root_dir
        self.sector = sector
        self.split = split
        self.transform = transform
        self.samples = []

        sector_path = os.path.join(root_dir, sector)
        if not os.path.exists(sector_path):
            # Fallback to searching subdirectories
            subdirs = glob.glob(os.path.join(root_dir, f"*{sector}*"))
            sector_path = subdirs[0] if subdirs else root_dir

        if split == "train":
            # Nominal pristine training samples only (One-class paradigm)
            norm_dir = os.path.join(sector_path, "normal")
            if os.path.exists(norm_dir):
                for f in glob.glob(os.path.join(norm_dir, "*.png")) + glob.glob(os.path.join(norm_dir, "*.jpg")):
                    self.samples.append((f, 0)) # 0: Normal
        else:
            # Test split: normal + all defect types
            for root, dirs, files in os.walk(sector_path):
                for f in files:
                    if f.endswith((".png", ".jpg", ".jpeg")):
                        full_path = os.path.join(root, f)
                        label = 0 if "normal" in root.lower() else 1
                        self.samples.append((full_path, label))

    def __len__(self):
        return max(1, len(self.samples))

    def __getitem__(self, idx):
        if len(self.samples) == 0:
            # Dummy tensor fallback for testing
            img_tensor = torch.zeros((3, 256, 256), dtype=torch.float32)
            return img_tensor, 0, "dummy.png"

        path, label = self.samples[idx % len(self.samples)]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

def get_dataloader(root_dir: str, sector: str, split: str = "train", batch_size: int = 16, image_size: int = 256):
    """Constructs and returns a PyTorch DataLoader."""
    if split == "train":
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    ds = IndustrialDataset(root_dir, sector, split, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)
