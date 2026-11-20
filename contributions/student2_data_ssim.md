# Individual Contribution Log: Student 2 (Data Engineer & Loss Function Specialist)
**Name:** [ Full Name - Student 2 ] | **Student ID:** [ Student ID - 2 ]  
**Project:** Industrial Visual Anomaly Detection (AI403) — Pharos University in Alexandria  
**Track Ownership:** Multi-Sector Data Pipeline & Custom Differentiable SSIM Loss  

---

### Key Responsibilities & Deliverables:
1. **Multi-Sector Data Engineering (`src/datasets/industrial_dataset.py`, `data/preprocess.py`):**
   - Curated multi-sector test benchmark spanning 4 domains: Electronics PCB, Machined Nuts, Welded Seams, Ceramic Tiles.
   - Built custom PyTorch `Dataset` and `DataLoader` pipelines with one-class training split (nominal only) and multi-class test splits.
   - Designed domain-specific data augmentation (flips, slight jitter) while avoiding destructive cropping to preserve spatial alignment.

2. **Custom Loss Function Implementation (Requirement #5):**
   - Implemented **Differentiable SSIM Loss** from scratch as a PyTorch `nn.Module` (`src/models/cae.py`).
   - Built 2D depthwise convolution with an 11x11 Gaussian kernel ($\sigma = 1.5$) to evaluate luminance, contrast, and structural correlation.
   - Formulated combined loss $\mathcal{L} = 0.5 \cdot \mathcal{L}_1 + 0.5 \cdot (1 - \text{SSIM})$.

3. **Defect Physics Analysis:**
   - Documented failure modes and defect morphologies: solder bridge short circuits, track breaks, metal stress cracks, weld porosity.

### Weekly Work Log:
- **Week 1-2:** Industrial defect dataset research and taxonomy categorization across 4 manufacturing sectors.
- **Week 3-4:** Mathematical formulation and derivation of custom 2D Gaussian SSIM loss module.
- **Week 5-6:** PyTorch DataLoader integration, normalization standardization, and data verification.
- **Week 7-8:** Comparative evaluation of MSE vs. SSIM reconstruction losses for autoencoder training.
