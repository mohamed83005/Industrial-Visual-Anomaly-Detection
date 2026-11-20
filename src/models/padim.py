"""
Model 2: Modern Pretrained Baseline — PaDiM (ResNet-18)
Deep Learning (AI403) - Pharos University in Alexandria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class PaDiM(nn.Module):
    """
    Patch Distribution Modeling using pretrained ResNet-18 features.
    """
    def __init__(self, pretrained: bool = True, num_selected_dims: int = 100):
        super(PaDiM, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        # Freeze weights
        for param in self.parameters():
            param.requires_grad = False

        self.num_selected_dims = num_selected_dims
        # Total dims: layer1(64) + layer2(128) + layer3(256) = 448
        np.random.seed(42)
        self.idx = np.random.choice(448, num_selected_dims, replace=False)

        self.mean = None
        self.inv_cov = None

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x) # [B, 64, 64, 64]
            f2 = self.layer2(f1) # [B, 128, 32, 32]
            f3 = self.layer3(f2) # [B, 256, 16, 16]

            # Upsample f2 and f3 to f1 resolution (64x64)
            f2_up = F.interpolate(f2, size=f1.shape[2:], mode="bilinear", align_corners=False)
            f3_up = F.interpolate(f3, size=f1.shape[2:], mode="bilinear", align_corners=False)

            # Concatenate multi-scale representations
            features = torch.cat([f1, f2_up, f3_up], dim=1) # [B, 448, 64, 64]
            features = features[:, self.idx, :, :] # [B, 100, 64, 64]
            return features

    def fit(self, train_loader, device="cpu"):
        """Computes spatial mean and inverse covariance matrix across nominal training set."""
        self.to(device)
        self.eval()
        feature_list = []

        with torch.no_grad():
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device)
                feats = self.extract_features(imgs)
                feature_list.append(feats.cpu())

        if len(feature_list) == 0:
            return

        all_features = torch.cat(feature_list, dim=0) # [N, 100, 64, 64]
        N, C, H, W = all_features.shape

        all_features = all_features.permute(2, 3, 1, 0).reshape(H * W, C, N) # [4096, 100, N]

        mean = torch.mean(all_features, dim=2) # [4096, 100]
        cov = torch.zeros(H * W, C, C)

        for i in range(H * W):
            diff = all_features[i] - mean[i].unsqueeze(1)
            cov[i] = torch.mm(diff, diff.t()) / (N - 1) + 0.01 * torch.eye(C) # Regularization

        self.mean = mean.to(device)
        self.inv_cov = torch.linalg.inv(cov).to(device)
