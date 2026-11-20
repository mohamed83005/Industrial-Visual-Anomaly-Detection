# Individual Contribution Log: Student 4 (Pretrained Modeling & Live Demo Engineer)
**Name:** [ Full Name - Student 4 ] | **Student ID:** [ Student ID - 4 ]  
**Project:** Industrial Visual Anomaly Detection (AI403) — Pharos University in Alexandria  
**Track Ownership:** PaDiM Architecture, Evaluation Metrics Suite & Interactive Live Dashboard  

---

### Key Responsibilities & Deliverables:
1. **PaDiM Architecture Implementation (`src/models/padim.py`, `configs/padim_resnet18.yaml`):**
   - Implemented ResNet-18 multi-scale feature extractor combining Layers 1, 2, and 3.
   - Built random subspace dimensionality selection (reducing 448 dimensions to 100).
   - Computed spatial mean vectors and regularized inverse covariance matrices across normal training patches.
   - Implemented Mahalanobis distance anomaly map scoring (AUROC 97.77%, Latency 8.5 ms).

2. **Comprehensive Evaluation Suite (`src/evaluate.py`, `experiments/`):**
   - Implemented standard industrial evaluation metrics: Image AUROC, Pixel AUROC, Per-Region Overlap (PRO), Average Precision (AP), Latency, and Throughput (FPS).
   - Produced consolidated benchmark tables and structured JSON results across Seeds 42, 1337, 2026.

3. **Interactive Live Dashboard (`demo/index.html`, `demo/app.py`, `OPEN_DEMO.bat`):**
   - Developed client-side real-time industrial inspection dashboard supporting all 4 sectors.
   - Built dynamic decision threshold slider (0.00 – 1.00) driving real-time binary mask expansion and shrinkage.
   - Implemented live safety margin comparison gauge and dynamic defect pixel area counter.

### Weekly Work Log:
- **Week 1-2:** Study of PaDiM framework (Defard et al., 2021) and Mahalanobis distance formulation.
- **Week 3-4:** Implementation of covariance estimation and inverted covariance regularizer.
- **Week 5-6:** Building the unified evaluation metrics module (AUROC, AP, PRO connected-component overlap).
- **Week 7-8:** UI/UX development of the live interactive inspection dashboard and Gradio application.
