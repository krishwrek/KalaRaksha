"""
KalaRaksha - End-to-End Pipeline Verification Test
Tests all IPA detectors, heatmap fusion, region localization, priority classification,
and report generation across sample assets.
"""

import os
import sys
import numpy as np

from utils.preprocessing import preprocess_heritage_image
from utils.crack_detection import detect_cracks
from utils.moisture_detection import detect_moisture_seepage
from utils.peeling_detection import detect_peeling_plaster
from utils.biological_growth import detect_biological_growth
from utils.erosion_detection import detect_surface_erosion
from utils.heatmap_fusion import fuse_deterioration_channels
from utils.region_analysis import analyze_damage_regions
from utils.priority_engine import classify_region_priority, compute_heritage_health_index
from utils.visualization import draw_damage_overlay, create_damage_analytics_plot
from utils.report_generator import generate_csv_report, generate_pdf_report


def test_full_pipeline():
    sample_path = "sample_images/sample1_temple_crack.jpg"
    print(f"Testing pipeline with {sample_path}...")

    # 1. Preprocessing
    prep = preprocess_heritage_image(sample_path, max_dim=1280, apply_clahe=True, gamma=1.0)
    print(f"  [OK] Preprocessed image: {prep['dimensions']}, mean luminance: {prep['metrics']['mean_luminance']}")

    gray = prep["gray"]
    rgb = prep["enhanced_rgb"]
    hsv = prep["hsv"]

    # 2. Crack detection
    cracks = detect_cracks(gray, sensitivity=0.6)
    print(f"  [OK] Crack detection: {cracks['num_crack_segments']} segments, length: {cracks['total_crack_length_px']}px, coverage: {cracks['coverage_pct']}%")

    # 3. Moisture detection
    moist = detect_moisture_seepage(rgb, hsv, sensitivity=0.5)
    print(f"  [OK] Moisture detection: {moist['num_patches']} patches, coverage: {moist['moisture_area_pct']}%")

    # 4. Peeling detection
    peel = detect_peeling_plaster(gray, rgb, sensitivity=0.5)
    print(f"  [OK] Peeling detection: {peel['num_peeling_patches']} patches, coverage: {peel['peeling_area_pct']}%")

    # 5. Biological growth
    bio = detect_biological_growth(rgb, hsv, sensitivity=0.5)
    print(f"  [OK] Biological growth: {bio['colony_count']} colonies, coverage: {bio['coverage_pct']}%")

    # 6. Erosion detection
    eros = detect_surface_erosion(gray, sensitivity=0.5)
    print(f"  [OK] Erosion detection: {eros['num_eroded_patches']} patches, coverage: {eros['eroded_area_pct']}%")

    # 7. Heatmap Fusion
    fusion = fuse_deterioration_channels(
        crack_density=cracks["crack_density"],
        moisture_density=moist["moisture_density"],
        peeling_density=peel["peeling_density"],
        biological_density=bio["biological_density"],
        erosion_density=eros["erosion_roughness"],
        enhanced_rgb=rgb,
        colormap_name="JET",
        alpha_blend=0.55
    )
    print(f"  [OK] Heatmap fusion: total damaged area = {fusion['total_damaged_pct']}%, active contributions: {fusion['channel_contributions']}")

    # 8. Region Analysis
    regions = analyze_damage_regions(
        composite_mask=fusion["composite_mask"],
        crack_mask=cracks["crack_mask"],
        moisture_mask=moist["moisture_mask"],
        peeling_mask=peel["peeling_mask"],
        biological_mask=bio["biological_mask"],
        erosion_mask=eros["erosion_mask"],
        fused_density=fusion["fused_density"],
        min_region_area=100
    )
    print(f"  [OK] Localized {len(regions)} damage region(s).")

    # 9. Priority Engine
    for reg in regions:
        pri_info = classify_region_priority(reg)
        reg.update(pri_info)

    health_metrics = compute_heritage_health_index(regions, fusion["total_damaged_pct"])
    print(f"  [OK] Heritage Health Index: {health_metrics['health_index']}/100, Priority: {health_metrics['overall_priority']} ({health_metrics['status_phrase']})")

    # 10. Visualization Overlay
    overlay = draw_damage_overlay(rgb, regions, draw_contours=True, draw_boxes=True, draw_labels=True)
    print(f"  [OK] Rendered damage overlay shape: {overlay.shape}")

    # 11. Plot generation
    fig = create_damage_analytics_plot(fusion["channel_contributions"], health_metrics["priority_counts"], health_metrics["health_index"])
    print(f"  [OK] Rendered analytics Matplotlib figure.")

    # 12. Report Exports
    csv_bytes = generate_csv_report(regions, health_metrics)
    print(f"  [OK] Generated CSV report ({len(csv_bytes)} bytes)")

    pdf_bytes = generate_pdf_report(
        original_rgb=prep["original_rgb"],
        enhanced_rgb=rgb,
        heatmap_rgb=fusion["heatmap_rgb"],
        overlay_rgb=overlay,
        regions=regions,
        overall_metrics=health_metrics,
        channel_contributions=fusion["channel_contributions"],
        image_filename="sample1_temple_crack.jpg"
    )
    print(f"  [OK] Generated PDF conservation dossier ({len(pdf_bytes)} bytes)")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("outputs/test_inventory.csv", "wb") as f:
        f.write(csv_bytes)
    print("  [SUCCESS] All pipeline checks passed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
