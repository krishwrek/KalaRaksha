# 🏛️ KalaRaksha: Heritage-Wall Deterioration Analysis & Restoration-Priority Mapping

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.15-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind-v4-38B2AC.svg)](https://tailwindcss.com/)
[![Cloudflare Pages](https://img.shields.io/badge/Deploy-Cloudflare%20Pages-F38020.svg)](https://pages.cloudflare.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **KalaRaksha** (काल रक्षा — *"Preserving the Legacy of Time"*) is an explainable **Image Processing Analysis (IPA)** system designed for the digital conservation of heritage temple walls, murals, historical cave frescoes, sculptures, and stone masonry.

---

## 🌟 Why Classical IPA Instead of Black-Box Deep Learning?

Many modern computer vision systems rely on opaque, GPU-hungry deep neural networks (YOLO, ResNet) that output a bounding box with an ungrounded confidence score. In heritage conservation, **this is insufficient**:
- **Conservators need scientific justifications:** Why was a patch classified as rising damp rather than an original iron-oxide mineral pigment?
- **Monuments lack GPUs in the field:** Rural archaeological sites require lightweight, deterministic, CPU-runnable pipelines.
- **Physical interpretability is paramount:** Interventions like hydraulic lime grouting or biocide application cannot be authorized without measurable physical criteria (crack width, edge density, Excess Green chlorophyll index).

**KalaRaksha** delivers deterministic, explainable conservation diagnostics where every detected region is backed by mathematical metrics and linked directly to **ICOMOS**, **ICCROM**, and **ASI** restoration charters.

---

## 🏗️ System Architecture & Analysis Pipeline

```
                     Upload Heritage Image (JPG / PNG / WEBP)
                                        │
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │  Preprocessing: Max 1280px Scale + LAB CLAHE + Bilateral Denoise │
      └───────────────────────────────────────────────────────────────────┘
                                        │
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
  [Crack Detection]             [Moisture Seepage]            [Peeling Plaster]
  - Morphological Black-Hat     - LAB Luminance Drop (ΔL)     - Multi-scale Sobel
  - Adaptive Canny Edges        - HSV Damp Color Bands        - Local Edge Density
  - Hough Line Transform        - Amoeboid Contour Compactness- Surface Discontinuity
  - Thinning Skeletonization    - Mineral Tide Lines          - Curling Boundary Relief
         │                              │                              │
         ├──────────────────────────────┴──────────────────────────────┤
         ▼                                                             ▼
  [Biological Growth]                                           [Surface Erosion]
  - Excess Green Index (ExG = 2g - r - b)                       - Local Variance (σ²)
  - Dual-band HSV Chlorophyll & Lichen                          - Local Shannon Entropy
  - Green Dominance Verification (G > R)                        - Laplacian Micro-Pitting
         │                                                             │
         └──────────────────────────────┬──────────────────────────────┘
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   Multi-Modal Weighted Heatmap Fusion     │
                  │   H_fused = Σ w_k · D_k (Gaussian Smooth) │
                  └───────────────────────────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    Connected-Component Damage Analysis    │
                  │    - Polygonal Contour Localization       │
                  │    - Primary / Compound Attribution       │
                  │    - Severity Score (0-100) Computation   │
                  └───────────────────────────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │      Restoration Priority Mapping         │
                  │  🔴 Critical | 🟠 High | 🟡 Moderate | 🟢 Low │
                  └───────────────────────────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │    Heritage Health Index (HHI / 100)      │
                  │  + Actionable Directives (ICOMOS/ASI)     │
                  └───────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
       [Interactive Streamlit UI]             [Downloadable Outputs]
       - Side-by-side Inspection              - Official PDF Dossier (ReportLab)
       - 5-Channel Decomposition              - CSV Damage Inventory (Pandas)
       - Statistical Analytics                - High-Res Annotated PNG Overlay
```

---

## 🎯 Core Capabilities & Detectors

| Deterioration Mode | Core Image Processing Analysis (IPA) Technique | Key Output Metrics |
| :--- | :--- | :--- |
| **⚡ Cracks & Fissures** | Multi-scale Black-Hat morphology, adaptive Canny, Probabilistic Hough Lines, Guo-Hall skeletonization | Crack length (px), branch count, mean width, elongation ratio |
| **💧 Moisture & Damp** | Background illumination subtraction in LAB space, brown/damp HSV band thresholding, amoeboid contour compactness | Damaged area (%), plume count, largest contiguous plume size |
| **🍂 Peeling & Flaking** | Sobel gradient magnitude, spatial box-filter edge density convolution, local variance step-change | Edge density index, flaking cluster count, boundary relief |
| **🌿 Biological Growth** | Spectroscopic Excess Green Index ($2g - r - b$), dual-band chlorophyll segmentation, green-over-red dominance | Coverage area (%), colony count, Excess Green mean |
| **🌪️ Surface Erosion** | Moving-window variance $\mathbb{E}[I^2] - (\mathbb{E}[I])^2$, local Shannon entropy, Laplacian roughness | Roughness score (/100), entropy dispersion, pitting index |

---

## 🎨 Restoration Priority Hierarchy

KalaRaksha employs standardized conservation color-coding across overlays, tables, and reports:

- 🔴 **CRITICAL (Red, `#E53935`):**
  - **Triggers:** Large moisture area (> 3.5%), deep/long structural cracks (aspect ratio > 6), severe plaster flaking (> 2.5%), or compound multi-hazard damage.
  - **Action Timeline:** Immediate emergency intervention required within 72 hours.
- 🟠 **HIGH (Orange, `#FB8C00`):**
  - **Triggers:** Multiple overlapping defect modes, medium cracks, active biological colonies.
  - **Action Timeline:** Conservation program to be scheduled within 30-60 days.
- 🟡 **MODERATE (Yellow, `#FDD835`):**
  - **Triggers:** Small hairline cracks, localized superficial dampness, early plaster blisters, granular roughness.
  - **Action Timeline:** Scheduled preventive maintenance within 6 months.
- 🟢 **LOW (Green, `#43A047`):**
  - **Triggers:** Minor surface anomalies, stable historic patina, or early micro-weathering.
  - **Action Timeline:** Routine annual monitoring and condition documentation.

---

## 📁 Repository Structure

```
kalaraksha/
├── app.py                      # Streamlit interactive conservation dashboard
├── api.py                      # Production FastAPI headless REST API
├── train_classifier.py         # Production training & validation pipeline
├── requirements.txt            # Python dependencies (Streamlit, FastAPI, scikit-learn, OpenCV)
├── Dockerfile                  # Lean multi-architecture production container
├── docker-compose.yml          # Dual-service orchestration (UI + REST API)
├── README.md                   # Project documentation & reference
├── create_sample_assets.py     # Procedural heritage asset generator
├── test_pipeline.py            # Automated end-to-end verification test suite
├── models/                     # Trained production ML model artifacts
│   ├── kalaraksha_damage_classifier.joblib
│   ├── scaler.joblib
│   ├── model_metadata.json
│   └── training_evaluation.png
├── utils/
│   ├── __init__.py             # Clean module exports
│   ├── preprocessing.py        # Max 1280px bounding, LAB CLAHE, Bilateral filtering
│   ├── crack_detection.py      # Black-Hat, adaptive Canny, Hough, skeletonization
│   ├── moisture_detection.py   # Relative darkness depression, damp HSV, amoeboid plumes
│   ├── peeling_detection.py    # Sobel gradient magnitude, local edge density, variance
│   ├── biological_growth.py    # Excess Green Index (ExG), HSV chlorophyll, colony clusters
│   ├── erosion_detection.py    # Local variance, Shannon entropy, Laplacian roughness
│   ├── heatmap_fusion.py       # Weighted multi-hazard fusion & thermal color maps
│   ├── region_analysis.py      # Connected components, contour localization, explainability
│   ├── ml_classifier.py        # Production hybrid IPA-ML classification & feature extraction
│   ├── priority_engine.py      # Heritage Health Index (HHI) & restoration priority engine
│   ├── report_generator.py     # Publication-quality PDF Dossier & CSV export
│   └── visualization.py        # Annotated overlays, bounding boxes, Matplotlib analytics
├── sample_images/              # 5 pre-packaged heritage wall sample photographs
├── outputs/                    # Output directory for generated PDF & CSV reports
└── docs/
    ├── ARCHITECTURE.md         # Mathematical derivations of all IPA algorithms
    ├── CONSERVATION_GUIDELINES.md # ICOMOS, ICCROM, and ASI restoration protocols
    └── PRODUCTION_DEPLOYMENT.md# Production deployment guide (Docker, Cloud Run, AWS, Offline)
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested up to Python 3.15)
- Standard commodity CPU (no GPU or CUDA needed)

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/kalaraksha.git
cd kalaraksha
pip install -r requirements.txt
```

### 3. Generate Preset Heritage Sample Assets
Generate 5 synthetic heritage wall photographs for testing:
```bash
python create_sample_assets.py
```

### 4. Run Verification Test Suites
Verify that all 5 IPA detectors, ML classification, and security endpoints operate with zero vulnerabilities:
```bash
python test_pipeline.py
python test_security.py
```

### 5. Launch Full-Stack Application

#### Step A: Launch Production FastAPI Backend
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Swagger API docs: `http://localhost:8000/docs`

#### Step B: Launch Independent React + Tailwind Frontend
```bash
cd frontend
npm install
npm run dev
```
Open your browser at `http://localhost:5173`.

*(Optional Legacy Streamlit Mode)*
```bash
streamlit run app.py
```

---

## ⛅ Cloudflare Deployment Guide

KalaRaksha's independent frontend is optimized for **Cloudflare Pages** static & edge hosting:

### 1. Build the Production Bundle
```bash
cd frontend
npm run build
```
This generates a minified, tree-shaken static bundle in `frontend/dist/` with edge security headers (`_headers`) and SPA routing (`_redirects`).

### 2. Deploy via Wrangler CLI
```bash
# Login to your Cloudflare account
npx wrangler login

# Deploy directly to Cloudflare Pages
npx wrangler pages deploy frontend/dist --project-name kalaraksha
```

### 3. Or Connect via GitHub CI/CD
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) > **Workers & Pages** > **Create application** > **Pages** > **Connect to Git**.
2. Select repository: `https://github.com/krishwrek/KalaRaksha`.
3. Configure build settings:
   - **Framework preset:** `Vite`
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
4. Click **Save and Deploy**. Every push to `main` will automatically build and deploy your site to `https://kalaraksha.pages.dev`!

---

## 🖼️ Authentic Heritage Benchmarks

The system includes 4 authentic Indian monument photographs ready for instant diagnostic benchmarking:
1. **🏛️ Konark Sun Temple (Deula, Odisha):** 13th-century Kalinga architecture, multi-tiered khondalite stone weathering.
2. **🛕 Golden Dravidian Gopuram:** Dravidian temple gateway tower with multi-tiered sculpted stucco relief.
3. **💧 Carved Subterranean Stepwell (Vav):** Western Indian subterranean masonry with moisture seepage bands and subterranean salt efflorescence.
4. **🏰 Ornate Rajasthani Sandstone Jharokha:** Intricate fretted sandstone lattice facade showing atmospheric patina and edge weathering.

---

## 📄 Downloadable Conservation Dossier

With one click, conservators can export:
1. **Official PDF Conservation Dossier:**
   - Document ID, timestamp, and asset resolution metadata
   - Overall Heritage Health Index (HHI) and Restoration Urgency rating
   - 4-Plate Photographic Evidence Sheet (Original, Enhanced, Fused Heatmap, Priority Overlay)
   - Quantitative Damage Inventory Table with full physical diagnostics
   - Actionable Conservation Directives aligned with international standards
   - Conservator and Superintending Archaeologist attestation sign-off block
2. **Machine-Readable CSV Inventory:**
   - Full tabular export containing Region ID, coordinates, area (px and %), primary damage type, severity score, priority tier, and recommendations.
3. **High-Resolution Annotated PNG Overlay:**
   - Overlay image containing color-coded bounding boxes, contour polygons, and badge tags.

---

## 📜 International Charters Compliance

All conservation directives provided by KalaRaksha adhere to:
- **ICOMOS Venice Charter (1964):** Minimality of intervention and preservation of authentic material fabric.
- **ICCROM Stone Conservation Guidelines:** Non-destructive diagnostics, compatible breathable consolidants (ethyl silicate / TEOS), and prohibition of impervious synthetic coatings.
- **Archaeological Survey of India (ASI) Principles:** Strict prohibition of Portland cement on historic lime/mud masonry.

---

## ⚖️ License & Citation

Distributed under the **MIT License**.

If you use KalaRaksha in research or heritage conservation projects, please cite:
```bibtex
@software{kalaraksha2026,
  author = {KalaRaksha Conservation Intelligence Team},
  title = {KalaRaksha: An IPA-Based Heritage-Wall Deterioration Analysis and Restoration-Priority Mapping System},
  year = {2026},
  url = {https://github.com/krishwrek/KalaRaksha}
}
```
