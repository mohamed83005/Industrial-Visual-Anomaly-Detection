"""
Model 1: Classic Convolutional Autoencoder (CAE) + Custom SSIM Loss
Deep Learning (AI403) - Pharos University in Alexandria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    """
    Custom Differentiable Structural Similarity (SSIM) Loss (Requirement #5).
    Implemented from scratch using an 11x11 Gaussian filter.
    """
    def __init__(self, window_size: int = 11, channels: int = 3, sigma: float = 1.5):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create 1D Gaussian kernel
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = (gauss / gauss.sum()).unsqueeze(1)
        
        # Create 2D separable Gaussian window
        window_2d = gauss.mm(gauss.t()).unsqueeze(0).unsqueeze(0)
        self.window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        device = img1.device
        window = self.window.to(device)

        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=self.channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return 1.0 - ssim_map.mean()

class ConvAutoencoder(nn.Module):
    """
    Classic 4-stage convolutional autoencoder baseline.
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, latent_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out

    def get_anomaly_map(self, x: torch.Tensor) -> torch.Tensor:
        recon = self.forward(x)
        diff = torch.abs(x - recon).mean(dim=1, keepdim=True)
        return diff
