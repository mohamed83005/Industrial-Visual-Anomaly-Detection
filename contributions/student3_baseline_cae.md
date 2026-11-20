# Individual Contribution Log: Student 3 (Baseline Deep Learning Engineer)
**Name:** [ Full Name - Student 3 ] | **Student ID:** [ Student ID - 3 ]  
**Project:** Industrial Visual Anomaly Detection (AI403) — Pharos University in Alexandria  
**Track Ownership:** Convolutional Autoencoder (CAE) Architecture & Multi-Seed Training  

---

### Key Responsibilities & Deliverables:
1. **Convolutional Autoencoder Architecture (`src/models/cae.py`):**
   - Designed 4-stage convolutional encoder (Conv2d, BatchNorm, LeakyReLU) downsampling from $256 \times 256$ to $16 \times 16$.
   - Engineered bottleneck latent dimension (256 channels) balancing compression with feature retention.
   - Designed 4-stage transposed convolutional decoder (ConvTranspose2d, BatchNorm, LeakyReLU) with Sigmoid output.

2. **Training & Multi-Seed Experimentation (`src/train.py`, `configs/cae_baseline.yaml`):**
   - Built training execution pipeline with Adam optimizer, learning rate scheduling, and checkpoint saving.
   - Evaluated training stability across multiple random seeds (Seeds 42, 1337, 2026).
   - Generated pixel-level reconstruction residual error maps.

3. **Failure Mode Analysis (Viva Defense):**
   - Conducted rigorous empirical investigation into why autoencoders fail (AUROC 84.17% vs 99.37%):
     - Edge blurriness causing false positives on nominal traces.
     - Identity shortcut phenomenon causing autoencoders to inadvertently reconstruct small hairline flaws.

### Weekly Work Log:
- **Week 1-2:** Baseline autoencoder architecture design and layer dimension calculations.
- **Week 3-4:** Implementation of encoder-decoder network in PyTorch with BatchNorm and LeakyReLU.
- **Week 5-6:** Training execution across nominal datasets; tuning bottleneck capacity and learning rates.
- **Week 7-8:** Multi-seed evaluation, reconstruction error map generation, and latency profiling (3.8 ms).
