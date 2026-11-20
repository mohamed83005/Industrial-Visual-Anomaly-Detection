# Multi-Sector Industrial Visual Anomaly Detection (Image → Anomaly Score + Heatmap)
**Pharos University in Alexandria (PUA) — Faculty of Computer Science & Artificial Intelligence**  
**Course:** Deep Learning (AI403) — Fall 2026  
**Supervision:** Dr. Sahar Ghanem  
**Teaching Assistants:** Eng. Esraa Habiba, Eng. Philopater Hany, Eng. Mohamed Ashraf  
**Team Name:** DeepDetect PUA  
**Selected Project Idea:** Idea #6 — Industrial Visual Anomaly Detection (Image → Anomaly Score + Map)  

---

## 📁 Repository Structure (Section 4.1 Specification)

```text
DL PROJECT/
│
├── README.md                              # Project documentation & execution guide
├── requirements.txt                       # Project Python dependencies
│
├── configs/                               # One configuration file per experiment
│   ├── cae_baseline.yaml                  # Classic CNN Autoencoder config
│   ├── padim_resnet18.yaml                # PaDiM (ResNet-18) config
│   └── patchcore_wrn50.yaml               # PatchCore (WideResNet-50-2) SOTA config
│
├── data/                                  # Download & preprocessing scripts (no raw data)
│   ├── download_mvtec.py                  # Dataset download utility
│   └── preprocess.py                      # Image normalization & transformation pipeline
│
├── src/                                   # Core source code
│   ├── datasets/                          # Loaders, splits, domain augmentation
│   │   ├── __init__.py
│   │   └── industrial_dataset.py          # Multi-sector dataset loader
│   ├── models/                            # One file per model family
│   │   ├── __init__.py
│   │   ├── cae.py                         # CAE architecture + custom SSIM loss
│   │   ├── padim.py                       # PaDiM Gaussian patch modeling
│   │   └── patchcore.py                   # PatchCore + custom Greedy K-Center Coreset
│   ├── train.py                           # Unified training entrypoint
│   ├── evaluate.py                        # Shared evaluation script for all models
│   └── industrial_anomaly_detection_pipeline.ipynb # Google Colab / Kaggle notebook
│
├── experiments/                           # Logs, result tables, figures
│   ├── benchmark_results.csv              # Official quantitative metrics CSV
│   └── experiment_summary.json            # Machine-readable multi-seed results
│
├── checkpoints/                           # Model weights & download links
│   └── README.md                          # Checkpoint download & generation instructions
│
├── demo/                                  # Interactive demonstration applications
│   ├── app.py                             # Local Gradio web server application (Port 7860)
│   └── index.html                         # Client-side live dashboard (instant zero-server launch)
│
├── report/                                # Technical reports & presentation slides
│   ├── Technical_Report_Industrial_Visual_Anomaly_Detection.docx # Word Technical Report
│   ├── One-inspection-system-four-industrial-worlds (1).pptx     # PowerPoint Presentation
│   ├── PROJECT_EXPLANATION_AND_MODEL_CHOICE.md                   # Full Architectural Documentation
│   └── GAMMA_PRESENTATION_PROMPT.md                              # 15-Slide Gamma AI Master Prompt
│
├── contributions/                         # Individual contribution logs (4 students)
│   ├── student1_lead_coreset.md           # Student 1: SOTA Lead & Coreset Engineer
│   ├── student2_data_ssim.md              # Student 2: Data Pipeline & SSIM Specialist
│   ├── student3_baseline_cae.md           # Student 3: Baseline Deep Learning Engineer
│   └── student4_padim_eval_demo.md        # Student 4: PaDiM, Evaluation & Live Demo
│
├── test_data_industrial/                  # 18 Multi-Sector Industrial Test Benchmark Images
│   ├── 01_electronics_pcb/                # Electronics: Normal, Solder Bridge, Broken Track
│   ├── 02_metal_machining_nut/            # Fasteners: Normal, Structural Crack, Scratch
│   ├── 03_metal_casting/                  # Welds: Normal, Transverse Crack, Porosity
│   └── 04_industrial_tile/                # Ceramics: Normal, Hairline Crack, Oil Blot
│
└── OPEN_DEMO.bat                          # One-click launcher for the live interactive demo
```

---

## ⚡ Quick Start Guide

### 1. Launch the Live Interactive Inspection Dashboard
- Double-click **`OPEN_DEMO.bat`** in the project root.
- Opens instantly in your browser (Chrome or Edge) with zero external server dependencies:
  - **4 Industrial Verticals:** PCB Circuits, Metal Fasteners, Heavy Welds, and Ceramic Tiles.
  - **3 Model Families:** PatchCore (SOTA), PaDiM (Pretrained), and CAE (Baseline).
  - **Dynamic Threshold Slider (0.00 – 1.00):** Segmented defect mask dynamically expands and shrinks in real time.
  - **Live Defect Pixel Counter & Safety Margin Gauge:** Displays exact anomalous pixel area and tolerance gap.

### 2. Run Training & Shared Evaluation
```bash
# Install dependencies
pip install -r requirements.txt

# Train or fit a specific architecture
python src/train.py --config configs/patchcore_wrn50.yaml
python src/train.py --config configs/padim_resnet18.yaml
python src/train.py --config configs/cae_baseline.yaml

# Run shared evaluation across models
python src/evaluate.py --model patchcore
python src/evaluate.py --model padim
python src/evaluate.py --model cae
```

---

## 📊 Cross-Architecture Benchmark Performance (MVTec AD)

| Deep Learning Architecture | Model Family | Image AUROC | Pixel AUROC | PRO Score | Inference Latency | Throughput (FPS) | Evaluation Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **⭐ PatchCore (WideResNet-50-2)** | **Advanced Memory Bank** | **99.37% ± 0.25%** | **98.60% ± 0.20%** | **94.80% ± 0.30%** | **22.4 ms** | **44.6 FPS** | **🥇 WINNER (Selected SOTA)** |
| **PaDiM (ResNet-18)** | Modern Pretrained | 97.77% ± 0.35% | 97.50% ± 0.20% | 92.10% ± 0.30% | 8.5 ms | 117.6 FPS | 🥈 Modern Fast Baseline |
| **Convolutional Autoencoder (CAE)** | Classic CNN Baseline | 84.17% ± 0.65% | 88.33% ± 0.41% | 72.43% ± 0.60% | 3.8 ms | 263.2 FPS | 🥉 Reconstruction Baseline |

---

## 👥 Four-Student Work Allocation & Viva Defense Structure

| Engineering Role | Assigned Student | Core Deliverables | Viva Defense Focus |
| :--- | :--- | :--- | :--- |
| **Student 1: SOTA Lead** | `[ Full Name - Student 1 ]` | PatchCore WRN50, custom Greedy K-Center Coreset written from scratch (90% compression), nearest-neighbor score derivation. | Minimax facility location complexity $\mathcal{O}(k \cdot N \cdot d)$, distance caching, memory bank subsampling. |
| **Student 2: Data & Loss Specialist** | `[ Full Name - Student 2 ]` | Multi-sector dataset loader, custom Differentiable SSIM loss module written from scratch (11x11 Gaussian window), data normalization. | SSIM vs MSE mathematical derivation, high-frequency edge penalty, spatial normalization invariance. |
| **Student 3: Baseline DL Engineer** | `[ Full Name - Student 3 ]` | 4-stage Convolutional Autoencoder (CAE), bottleneck latent dimension design, multi-seed training loops (Seeds 42, 1337, 2026). | Autoencoder reconstruction blur, identity shortcut failure mode, latent space compression. |
| **Student 4: Pretrained & Demo Engineer** | `[ Full Name - Student 4 ]` | PaDiM ResNet-18 modeling, multivariate Gaussian covariance estimation, Mahalanobis distance, interactive live web dashboard. | PRO vs Pixel AUROC connected component weighting, live dynamic threshold segmentation. |
