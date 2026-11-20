"""
Data Preprocessing and Augmentation Pipeline
Pharos University in Alexandria (AI403) - Deep Learning
"""

import os
from PIL import Image
import torchvision.transforms as T

def get_transforms(image_size: int = 256, is_training: bool = True):
    """
    Returns image transformation pipeline.
    Avoids spatial cropping to preserve nominal layout for industrial assemblies.
    """
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.ColorJitter(brightness=0.05, contrast=0.05),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def preprocess_image(image_path: str, image_size: int = 256):
    """Preprocesses a single image for model inference."""
    img = Image.open(image_path).convert("RGB")
    transform = get_transforms(image_size, is_training=False)
    return transform(img).unsqueeze(0)

if __name__ == "__main__":
    print("[+] Preprocessing pipeline loaded successfully.")
