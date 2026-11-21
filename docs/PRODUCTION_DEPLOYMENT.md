# KalaRaksha: Production Training & Deployment Guide

## 1. Production Architecture Overview

**KalaRaksha** uses a **Hybrid IPA-ML (Image Processing Analysis + Machine Learning)** architecture engineered for high-reliability, explainable heritage conservation:

```
[Raw Heritage Wall Image]
           │
           ▼
[Classical IPA Feature Detectors] ─── Deterministic, physically explainable
  - LAB CLAHE & Bilateral Filter
  - Morphological Black-Hat + Canny
  - HSV/LAB Luminance Depression & Damp Chromaticity
  - Sobel Gradient Energy & Spatial Edge Density
  - Excess Green Vegetation Index (ExG)
  - Moving-Window Variance & Shannon Entropy
           │
           ▼
[20-Dimensional Physical Feature Vector]
  (Aspect Ratio, Compactness, Luminance Drop, Edge Density, ExG, Sobel Energy, etc.)
           │
           ▼
[Calibrated Production ML Classifier (Random Forest / GBDT)]
  - Evaluates feature vector against learned statistical distributions
  - Outputs Calibrated Class Probabilities: P(Crack), P(Moisture), P(Peeling), etc.
  - Predicts Continuous Severity Score [0 - 100]
           │
           ▼
[Restoration Priority Engine + Actionable ICOMOS/ASI Directives]
           │
           ▼
[Multi-Channel Deployment: Web UI (Streamlit) & Headless REST API (FastAPI)]
```

---

## 2. Model Training & Retraining (`train_classifier.py`)

### 2.1 Training on Synthetic & Augmented Physical Features
To train or retrain the production model:
```bash
python train_classifier.py
```
This executes:
1. Generation of 1,200 multi-class physical feature vectors with randomized lighting, stone texture noise, and measurement jitter.
2. Stratified 5-Fold Cross-Validation.
3. Random Forest training with class-balanced weighting (`n_estimators=120`, `max_depth=12`).
4. Hold-out test evaluation generating a confusion matrix and classification report.
5. Export of production artifacts:
   - `models/kalaraksha_damage_classifier.joblib`
   - `models/scaler.joblib`
   - `models/model_metadata.json`
   - `models/training_evaluation.png`

### 2.2 Retraining on Real Heritage Wall Datasets
When field archaeological teams collect real annotated photographs (e.g. from ASI, UNESCO, or university archives):
1. Place cropped defect patches in `data/train/<class_name>/` (classes: `crack`, `moisture`, `peeling`, `biological`, `erosion`, `compound`).
2. Run the feature extraction script to convert real image patches into feature matrices using `utils.ml_classifier.extract_region_features`.
3. Fit the production model:
```python
from utils.ml_classifier import extract_region_features
import joblib

# Ingest real features and retrain
rf.fit(X_real_scaled, y_real)
joblib.dump(rf, "models/kalaraksha_damage_classifier.joblib")
```

---

## 3. Local Container Deployment (Docker & Docker Compose)

### 3.1 Single Container Run (Streamlit UI)
Build the production Docker image:
```bash
docker build -t kalaraksha:latest .
```
Run the interactive Streamlit UI:
```bash
docker run -d -p 8501:8501 --name kalaraksha-app kalaraksha:latest
```
Access the application at `http://localhost:8501`.

### 3.2 Dual-Service Deployment (UI + FastAPI REST Server)
Using `docker-compose.yml`, both the Streamlit UI and the high-throughput FastAPI server can be deployed concurrently:
```bash
docker-compose up -d --build
```
- **Streamlit UI:** `http://localhost:8501`
- **FastAPI REST API:** `http://localhost:8000`
- **Swagger Interactive API Docs:** `http://localhost:8000/docs`

---

## 4. Cloud Deployment Targets

### 4.1 Google Cloud Run (Serverless Container)
Cloud Run is ideal because KalaRaksha requires no GPU, cold-starts quickly, and scales to zero when idle:
```bash
# Authenticate with GCP
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]

# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/[YOUR_PROJECT_ID]/kalaraksha:latest
gcloud run deploy kalaraksha-service \
    --image gcr.io/[YOUR_PROJECT_ID]/kalaraksha:latest \
    --platform managed \
    --region asia-south1 \
    --memory 2Gi \
    --cpu 2 \
    --allow-unauthenticated
```

### 4.2 AWS Elastic Container Service (ECS) with Fargate
1. Push `kalaraksha:latest` to AWS ECR.
2. Create an ECS Task Definition with:
   - Compatibility: `FARGATE`
   - CPU: `1024` (1 vCPU)
   - Memory: `2048` (2 GB)
   - Port Mappings: `8501` (UI) or `8000` (API)
3. Attach an Application Load Balancer (ALB) pointing to the target group.

### 4.3 Hugging Face Spaces / Streamlit Community Cloud
1. Push this repository to a public or private GitHub repository.
2. Link to Streamlit Cloud or Hugging Face Spaces (select SDK: Streamlit, Python: 3.10+).
3. The application will build automatically using `requirements.txt` and launch `app.py`.

---

## 5. Offline / Air-Gapped Field Deployment

Many historical monuments, cave temples, and remote excavation sites lack reliable cellular or internet connectivity.

**KalaRaksha is 100% self-contained and operates entirely offline:**
1. Package the repository onto an external SSD or field laptop.
2. Install dependencies via cached wheel archive:
   ```bash
   pip install --no-index --find-links=./wheels -r requirements.txt
   ```
3. Run locally:
   ```bash
   streamlit run app.py
   ```
4. Field conservators can plug in DSLR or drone SD cards, run complete multi-hazard diagnoses, and generate official PDF reports on-site without internet access.

---

## 6. Production REST API Reference

### 6.1 Analyze Wall Image (`POST /api/v1/analyze`)
**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/analyze?sensitivity=0.55&enable_cracks=true" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@sample_images/sample1_temple_crack.jpg"
```

**Response (JSON):**
```json
{
  "status": "success",
  "filename": "sample1_temple_crack.jpg",
  "timestamp": "2026-11-21T14:20:00",
  "dimensions": { "width": 960, "height": 720 },
  "overall_health_index": 2.0,
  "restoration_priority": "CRITICAL",
  "urgency_phrase": "Emergency Conservation Required",
  "total_damaged_area_pct": 21.48,
  "detected_regions_count": 59,
  "regions": [
    {
      "region_id": "REG-01",
      "primary_damage": "crack",
      "primary_damage_name": "Structural / Hairline Crack",
      "priority": "CRITICAL",
      "severity_score": 88.5,
      "area_pct": 3.84,
      "ml_inference": {
        "predicted_class": "crack",
        "display_name": "Structural / Hairline Crack",
        "confidence": 0.962,
        "probabilities": { "crack": 0.962, "moisture": 0.012, "peeling": 0.015, "biological": 0.0, "erosion": 0.011, "compound": 0.0 }
      },
      "diagnostic_explanation": "Crack fracture with 78% regional dominance...",
      "recommendations": [
        "Install calibrated tell-tale glass or digital caliper gauges...",
        "Prohibit rigid Portland cement filling..."
      ]
    }
  ]
}
```

### 6.2 Export Official PDF Dossier (`POST /api/v1/report/pdf`)
```bash
curl -X POST "http://localhost:8000/api/v1/report/pdf" \
     -F "file=@sample_images/sample1_temple_crack.jpg" \
     --output Conservation_Report.pdf
```
