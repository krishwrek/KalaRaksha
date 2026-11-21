"""
KalaRaksha - Production FastAPI REST Server (Hardened Enterprise Edition)
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

Security Features:
1. Strict CORS configuration (configurable via ALLOWED_ORIGINS)
2. HTTP Security Headers (CSP, HSTS, X-Content-Type-Options, X-Frame-Options)
3. Sliding-Window Rate Limiting per IP
4. Payload magic-byte validation & decompression bomb prevention
5. Constant-time API Key authentication (via X-API-Key)
6. Masked error responses with correlation IDs to prevent internal info leakage
"""

import os
import io
import uuid
import base64
import logging
import datetime
from typing import Optional, Dict, Any, List
import numpy as np
import cv2
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Query, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import KalaRaksha Core Engines
from utils.preprocessing import preprocess_heritage_image
from utils.crack_detection import detect_cracks
from utils.moisture_detection import detect_moisture_seepage
from utils.peeling_detection import detect_peeling_plaster
from utils.biological_growth import detect_biological_growth
from utils.erosion_detection import detect_surface_erosion
from utils.heatmap_fusion import fuse_deterioration_channels
from utils.region_analysis import analyze_damage_regions
from utils.priority_engine import classify_region_priority, compute_heritage_health_index
from utils.visualization import draw_damage_overlay
from utils.report_generator import generate_csv_report, generate_pdf_report
from utils.ml_classifier import HeritageDamageClassifier
from utils.security import (
    validate_image_payload,
    sanitize_filename,
    InMemoryRateLimiter,
    verify_api_key,
    get_production_security_headers
)

# Configure secure server logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("KalaRakshaAPI")

# Environment & Security Setup
ENVIRONMENT = os.getenv("ENVIRONMENT", "production").lower()
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
rate_limiter = InMemoryRateLimiter(requests_per_minute=RATE_LIMIT_RPM)

# Parse allowed CORS origins
raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
allowed_origins = [orig.strip() for orig in raw_origins.split(",") if orig.strip()]

app = FastAPI(
    title="KalaRaksha Heritage Conservation API",
    description="Production-grade Image Processing Analysis (IPA) & Restoration Priority API for Heritage Monuments",
    version="1.0.0",
    docs_url="/docs" if ENVIRONMENT != "production" else "/docs",  # Kept accessible or configurable
    redoc_url=None
)

# Configure restricted CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_rate_limit_middleware(request: Request, call_next):
    """
    Inject production security headers and enforce IP rate limiting.
    """
    client_ip = request.client.host if request.client else "127.0.0.1"
    
    # Exclude basic healthcheck from rate limits
    if request.url.path not in ("/health", "/"):
        allowed, remaining = rate_limiter.is_allowed(client_ip)
        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too Many Requests",
                    "detail": "Rate limit threshold reached. Please wait before submitting additional analysis requests."
                },
                headers=get_production_security_headers()
            )

    response = await call_next(request)

    # Attach all hardened security headers
    for h_name, h_val in get_production_security_headers().items():
        response.headers[h_name] = h_val

    return response


classifier = HeritageDamageClassifier(model_dir="models")


def _np_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 85) -> str:
    """Encode numpy RGB array to base64 JPEG string."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


@app.get("/")
def root():
    return {
        "system": "KalaRaksha Heritage Conservation Intelligence",
        "status": "online",
        "documentation": "/docs",
        "version": "1.0.0",
        "environment": ENVIRONMENT,
        "ml_engine_loaded": classifier.is_loaded
    }


@app.get("/health")
def healthcheck():
    """Liveness & readiness probe."""
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now().isoformat(),
        "model_loaded": classifier.is_loaded,
        "model_metadata": {
            "model_name": classifier.metadata.get("model_name"),
            "version": classifier.metadata.get("version"),
            "test_accuracy": classifier.metadata.get("metrics", {}).get("test_accuracy")
        } if classifier.is_loaded and classifier.metadata else None
    }


@app.post("/api/v1/analyze")
async def analyze_wall(
    file: UploadFile = File(..., description="Wall photograph (JPG, PNG, WEBP)"),
    sensitivity: float = Query(0.55, ge=0.10, le=1.00, description="Detection sensitivity threshold"),
    enable_cracks: bool = Query(True, description="Enable structural & hairline crack detector"),
    enable_moisture: bool = Query(True, description="Enable moisture stains & seepage detector"),
    enable_peeling: bool = Query(True, description="Enable peeling paint & plaster detector"),
    enable_biological: bool = Query(True, description="Enable biological growth & moss detector"),
    enable_erosion: bool = Query(True, description="Enable surface erosion & weathering detector"),
    apply_clahe: bool = Query(True, description="Apply LAB CLAHE local contrast enhancement"),
    colormap: str = Query("JET", description="Heatmap color palette (JET, TURBO, INFERNO)"),
    return_images: bool = Query(True, description="Include base64 rendered overlay and heatmap"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key", description="Optional authentication key")
):
    """
    Execute full IPA + ML multi-hazard diagnosis on an uploaded heritage wall image.
    Protected by rate limiting, payload magic-byte checks, and decompression bomb guards.
    """
    # 1. Verify API Key if required
    if not verify_api_key(x_api_key):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing X-API-Key."
        )

    correlation_id = str(uuid.uuid4())

    try:
        # 2. Read and validate payload
        contents = await file.read()
        is_valid, safe_filename, err_msg = validate_image_payload(contents, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Validation failed: {err_msg}")

        # 3. Preprocess Image
        prep_data = preprocess_heritage_image(
            contents,
            max_dim=1280,
            apply_clahe=apply_clahe,
            filter_type="bilateral"
        )
        orig_rgb = prep_data["original_rgb"]
        enh_rgb = prep_data["enhanced_rgb"]
        gray = prep_data["gray"]
        hsv = prep_data["hsv"]
        lab = prep_data["lab"]
        h_img, w_img = prep_data["dimensions"]

        # 4. Execute Selected IPA Detectors
        crack_res = detect_cracks(gray, sensitivity=sensitivity) if enable_cracks else {
            "crack_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "crack_density": np.zeros((h_img, w_img), dtype=np.float32),
            "coverage_pct": 0.0, "total_crack_length_px": 0, "num_crack_segments": 0, "severity_score": 0.0,
            "explanation": "Disabled"
        }

        moist_res = detect_moisture_seepage(enh_rgb, hsv, sensitivity=sensitivity) if enable_moisture else {
            "moisture_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "moisture_density": np.zeros((h_img, w_img), dtype=np.float32),
            "moisture_area_pct": 0.0, "num_patches": 0, "severity_score": 0.0, "explanation": "Disabled"
        }

        peel_res = detect_peeling_plaster(gray, enh_rgb, sensitivity=sensitivity) if enable_peeling else {
            "peeling_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "peeling_density": np.zeros((h_img, w_img), dtype=np.float32),
            "peeling_area_pct": 0.0, "num_peeling_patches": 0, "severity_score": 0.0, "explanation": "Disabled"
        }

        bio_res = detect_biological_growth(enh_rgb, hsv, sensitivity=sensitivity) if enable_biological else {
            "biological_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "biological_density": np.zeros((h_img, w_img), dtype=np.float32),
            "coverage_pct": 0.0, "colony_count": 0, "severity_score": 0.0, "explanation": "Disabled"
        }

        eros_res = detect_surface_erosion(gray, sensitivity=sensitivity) if enable_erosion else {
            "erosion_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "erosion_roughness": np.zeros((h_img, w_img), dtype=np.float32),
            "eroded_area_pct": 0.0, "num_eroded_patches": 0, "severity_score": 0.0, "explanation": "Disabled"
        }

        # 5. Weighted Multi-Hazard Fusion
        toggles = {
            "cracks": enable_cracks,
            "moisture": enable_moisture,
            "peeling": enable_peeling,
            "biological": enable_biological,
            "erosion": enable_erosion
        }
        fusion = fuse_deterioration_channels(
            crack_density=crack_res["crack_density"],
            moisture_density=moist_res["moisture_density"],
            peeling_density=peel_res["peeling_density"],
            biological_density=bio_res["biological_density"],
            erosion_density=eros_res["erosion_roughness"],
            channel_toggles=toggles,
            enhanced_rgb=enh_rgb,
            colormap_name=colormap,
            alpha_blend=0.55
        )

        # 6. Region Localization & ML Feature Extraction
        regions = analyze_damage_regions(
            composite_mask=fusion["composite_mask"],
            crack_mask=crack_res["crack_mask"],
            moisture_mask=moist_res["moisture_mask"],
            peeling_mask=peel_res["peeling_mask"],
            biological_mask=bio_res["biological_mask"],
            erosion_mask=eros_res["erosion_mask"],
            fused_density=fusion["fused_density"],
            min_region_area=150,
            rgb_image=enh_rgb,
            gray_image=gray,
            hsv_image=hsv,
            lab_image=lab
        )

        # 7. Restoration Priority Classification
        for reg in regions:
            pri_data = classify_region_priority(reg)
            reg.update(pri_data)

        health_metrics = compute_heritage_health_index(regions, fusion["total_damaged_pct"])

        # Clean serializable regions
        clean_regions = []
        for reg in regions:
            clean_regions.append({
                "region_id": reg.get("region_id"),
                "primary_damage": reg.get("primary_damage"),
                "primary_damage_name": reg.get("primary_damage_name"),
                "priority": reg.get("priority"),
                "severity_score": reg.get("severity_score"),
                "area_pct": reg.get("area_pct"),
                "bbox": reg.get("bbox"),
                "centroid": reg.get("centroid"),
                "is_compound": reg.get("is_compound"),
                "channel_overlaps": reg.get("channel_overlaps"),
                "ml_inference": reg.get("ml_inference"),
                "diagnostic_explanation": reg.get("diagnostic_explanation"),
                "recommendations": reg.get("recommendations")
            })

        response_payload = {
            "status": "success",
            "correlation_id": correlation_id,
            "filename": safe_filename,
            "timestamp": datetime.datetime.now().isoformat(),
            "dimensions": {"width": w_img, "height": h_img},
            "overall_health_index": health_metrics["health_index"],
            "restoration_priority": health_metrics["overall_priority"],
            "urgency_phrase": health_metrics["status_phrase"],
            "total_damaged_area_pct": health_metrics["total_damaged_pct"],
            "channel_contributions": fusion["channel_contributions"],
            "priority_counts": health_metrics["priority_counts"],
            "detected_regions_count": len(clean_regions),
            "regions": clean_regions
        }

        # Include base64 images if requested
        if return_images:
            overlay_rgb = draw_damage_overlay(enh_rgb, regions, draw_contours=True, draw_boxes=True, draw_labels=True)
            response_payload["visualizations"] = {
                "annotated_overlay_base64": _np_to_base64_jpeg(overlay_rgb),
                "heatmap_blended_base64": _np_to_base64_jpeg(fusion["blended_rgb"])
            }

        return JSONResponse(content=response_payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CorrelationID: {correlation_id}] Unexpected error in analyze_wall: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while processing the heritage image. Reference Correlation ID: {correlation_id}"
        )


@app.post("/api/v1/report/pdf")
async def export_pdf(
    file: UploadFile = File(...),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """Generate and stream an official ReportLab PDF Conservation Dossier."""
    if not verify_api_key(x_api_key):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid or missing X-API-Key.")

    contents = await file.read()
    is_valid, safe_filename, err_msg = validate_image_payload(contents, file.filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Validation failed: {err_msg}")

    prep_data = preprocess_heritage_image(contents, max_dim=1280)
    enh_rgb = prep_data["enhanced_rgb"]
    gray = prep_data["gray"]
    hsv = prep_data["hsv"]
    lab = prep_data["lab"]

    c = detect_cracks(gray)
    m = detect_moisture_seepage(enh_rgb, hsv)
    p = detect_peeling_plaster(gray, enh_rgb)
    b = detect_biological_growth(enh_rgb, hsv)
    e = detect_surface_erosion(gray)

    fusion = fuse_deterioration_channels(
        c["crack_density"], m["moisture_density"], p["peeling_density"],
        b["biological_density"], e["erosion_roughness"], enhanced_rgb=enh_rgb
    )

    regions = analyze_damage_regions(
        fusion["composite_mask"], c["crack_mask"], m["moisture_mask"],
        p["peeling_mask"], b["biological_mask"], e["erosion_mask"],
        fusion["fused_density"], rgb_image=enh_rgb, gray_image=gray, hsv_image=hsv, lab_image=lab
    )

    for reg in regions:
        reg.update(classify_region_priority(reg))

    health_metrics = compute_heritage_health_index(regions, fusion["total_damaged_pct"])
    overlay_rgb = draw_damage_overlay(enh_rgb, regions)

    pdf_bytes = generate_pdf_report(
        original_rgb=prep_data["original_rgb"],
        enhanced_rgb=enh_rgb,
        heatmap_rgb=fusion["heatmap_rgb"],
        overlay_rgb=overlay_rgb,
        regions=regions,
        overall_metrics=health_metrics,
        channel_contributions=fusion["channel_contributions"],
        image_filename=safe_filename
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=KalaRaksha_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"}
    )
