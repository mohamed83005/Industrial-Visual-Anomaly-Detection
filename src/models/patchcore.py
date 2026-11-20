"""
Model 3: Advanced SOTA Winner — PatchCore (WideResNet-50-2)
Deep Learning (AI403) - Pharos University in Alexandria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class GreedyKCenterCoreset:
    """
    Custom Greedy K-Center Coreset Algorithm (Requirement #5).
    Implemented from scratch using vectorized NumPy distance matrices.
    Compresses nominal memory bank by 90% while retaining full coverage radius.
    """
    def __init__(self, subsample_ratio: float = 0.10):
        self.subsample_ratio = subsample_ratio

    def fit_sample(self, features: np.ndarray) -> np.ndarray:
        """
        Solves minimax facility location:
        arg min_{S} max_{m} min_{s in S} ||m - s||_2
        """
        n_samples = features.shape[0]
        n_coreset = max(1, int(n_samples * self.subsample_ratio))

        if n_samples <= n_coreset:
            return features

        # Initialize with first center (random)
        centers_idx = [np.random.randint(0, n_samples)]
        
        # Initial distance from all points to the first center
        first_center = features[centers_idx[0]:centers_idx[0]+1]
        dist_matrix = np.linalg.norm(features - first_center, axis=1)

        for _ in range(1, n_coreset):
            # Select point with maximum distance to its closest existing center
            new_center_idx = np.argmax(dist_matrix)
            centers_idx.append(new_center_idx)

            # Update distance matrix with minimum to existing centers
            new_center = features[new_center_idx:new_center_idx+1]
            new_dist = np.linalg.norm(features - new_center, axis=1)
            dist_matrix = np.minimum(dist_matrix, new_dist)

        return features[centers_idx]

class PatchCore(nn.Module):
    """
    PatchCore anomaly detection architecture using WideResNet-50-2 embeddings.
    """
    def __init__(self, subsample_ratio: float = 0.10):
        super(PatchCore, self).__init__()
        wrn = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.DEFAULT)
        
        self.conv1 = wrn.conv1
        self.bn1 = wrn.bn1
        self.relu = wrn.relu
        self.maxpool = wrn.maxpool
        self.layer1 = wrn.layer1
        self.layer2 = wrn.layer2
        self.layer3 = wrn.layer3

        for p in self.parameters():
            p.requires_grad = False

        self.coreset_sampler = GreedyKCenterCoreset(subsample_ratio=subsample_ratio)
        self.memory_bank = None

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            f2 = self.layer2(x) # [B, 512, 32, 32]
            f3 = self.layer3(f2) # [B, 1024, 16, 16]

            # Adaptive locally aware neighborhood pooling (3x3, stride 1)
            pool = nn.AvgPool2d(3, stride=1, padding=1)
            f2_pooled = pool(f2)
            f3_pooled = pool(f3)

            # Interpolate f3 to f2 resolution
            f3_up = F.interpolate(f3_pooled, size=f2.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate layers 2 and 3
            features = torch.cat([f2_pooled, f3_up], dim=1) # [B, 1536, 32, 32]
            return features

    def fit(self, train_loader, device="cpu"):
        """Extracts features from nominal training set and applies Greedy Coreset compression."""
        self.to(device)
        self.eval()
        patches_list = []

        with torch.no_grad():
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device)
                feats = self.extract_patch_features(imgs) # [B, 1536, 32, 32]
                B, C, H, W = feats.shape
                patches = feats.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
                patches_list.append(patches)

        if len(patches_list) > 0:
            all_patches = np.concatenate(patches_list, axis=0)
            print(f"[*] Total extracted patch features: {all_patches.shape[0]}")
            self.memory_bank = self.coreset_sampler.fit_sample(all_patches)
            print(f"[*] Compressed Coreset memory bank: {self.memory_bank.shape[0]} (90% compression)")
