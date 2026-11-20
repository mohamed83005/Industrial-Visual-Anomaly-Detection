# Model Checkpoints & Pretrained Weights
Pharos University in Alexandria (AI403) - Deep Learning

### Available Pretrained Checkpoints:
1. **PatchCore (WideResNet-50-2 Memory Bank):**
   - Pretrained ImageNet feature weights are automatically retrieved via `torchvision.models.wide_resnet50_2(weights=DEFAULT)`.
   - Compressed Greedy K-Center Coreset embeddings (~85 MB) are generated via `python src/train.py --config configs/patchcore_wrn50.yaml`.

2. **PaDiM (ResNet-18 Distribution Parameters):**
   - Pretrained ResNet-18 weights retrieved via `torchvision.models.resnet18(weights=DEFAULT)`.
   - Fitted spatial mean and inverse covariance matrices (~140 MB) saved upon running `python src/train.py --config configs/padim_resnet18.yaml`.

3. **Convolutional Autoencoder (CAE):**
   - Trained model weights (`cae_best.pth`, ~18 MB) saved automatically to `./checkpoints/cae_best.pth`.
