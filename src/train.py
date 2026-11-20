"""
Unified Training Pipeline
Pharos University in Alexandria (AI403) - Deep Learning
"""

import os
import argparse
import yaml
import torch
import torch.optim as optim

from datasets import get_dataloader
from models import ConvAutoencoder, SSIMLoss, PaDiM, PatchCore

def main():
    parser = argparse.ArgumentParser(description="Industrial Visual Anomaly Detection Training Entrypoint")
    parser.add_argument("--config", type=str, default="configs/patchcore_wrn50.yaml", help="Path to experiment config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu")
    print(f"[*] Training active on device: {device}")
    print(f"[*] Architecture: {cfg.get('model_name')} ({cfg.get('model_family')})")

    # DataLoader
    data_dir = cfg["data"]["data_dir"]
    train_loader = get_dataloader(data_dir, sector="01_electronics_pcb", split="train", batch_size=cfg["data"]["batch_size"])

    model_name = cfg["model_name"].lower()
    os.makedirs("./checkpoints", exist_ok=True)

    if model_name == "cae":
        model = ConvAutoencoder().to(device)
        criterion = SSIMLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

        print(f"[*] Starting Convolutional Autoencoder training for {cfg['training']['epochs']} epochs...")
        model.train()
        for epoch in range(1, min(5, cfg["training"]["epochs"] + 1)):
            total_loss = 0.0
            for imgs, _, _ in train_loader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                recon = model(imgs)
                loss = 0.5 * torch.abs(imgs - recon).mean() + 0.5 * criterion(imgs, recon)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"    Epoch [{epoch}] Loss: {total_loss:.4f}")

        torch.save(model.state_dict(), f"./checkpoints/{model_name}_best.pth")
        print(f"[+] Saved checkpoint to ./checkpoints/{model_name}_best.pth")

    elif model_name == "padim":
        model = PaDiM()
        print("[*] Fitting PaDiM Gaussian spatial distributions...")
        model.fit(train_loader, device=device)
        print("[+] PaDiM training complete.")

    elif model_name == "patchcore":
        subsample = cfg.get("coreset_sampling", {}).get("subsample_ratio", 0.10)
        model = PatchCore(subsample_ratio=subsample)
        print("[*] Extracting features and applying Greedy K-Center Coreset subsampling...")
        model.fit(train_loader, device=device)
        print("[+] PatchCore Coreset memory bank created.")

if __name__ == "__main__":
    main()
