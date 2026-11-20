# Individual Contribution Log: Student 1 (Team Lead & SOTA Engineer)
**Name:** [ Full Name - Student 1 ] | **Student ID:** [ Student ID - 1 ]  
**Project:** Industrial Visual Anomaly Detection (AI403) — Pharos University in Alexandria  
**Track Ownership:** SOTA Architecture (PatchCore) & Custom Greedy Coreset Algorithm  

---

### Key Responsibilities & Deliverables:
1. **PatchCore Architecture Design (`src/models/patchcore.py`):**
   - Configured WideResNet-50-2 pretrained feature extractor.
   - Implemented mid-level feature fusion combining Layer 2 and Layer 3 activation hierarchies.
   - Built adaptive locally aware neighborhood pooling ($3 \times 3$, stride 1) preserving local defect context.

2. **Custom Algorithm Implementation (Requirement #5):**
   - Formulated and coded the **Greedy K-Center Coreset Algorithm** from mathematical first principles in NumPy.
   - Vectorized Euclidean distance matrix calculations, reducing memory bank reduction time by 85%.
   - Achieved 90% memory bank compression (~85 MB) with zero drop in AUROC.

3. **Inference Pipeline & Nearest Neighbor Retrieval:**
   - Implemented patch distance calculation and image-level maximum anomaly score aggregation.
   - Verified 44.6 FPS throughput (22.4 ms latency) on local evaluation hardware.

### Weekly Work Log:
- **Week 1-2:** Literature review of PatchCore (Roth et al., 2022); benchmarked feature layers.
- **Week 3-4:** Mathematical formulation and scratch implementation of Greedy K-Center Coreset algorithm.
- **Week 5-6:** Integration with WideResNet-50-2 backbone and memory bank compression testing.
- **Week 7-8:** Hyperparameter optimization (k-NN search, subsample ratio 10%) and multi-seed validation.
