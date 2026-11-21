"""
KalaRaksha - Region Localization & Multi-Modal Damage Attribution Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module localizes discrete damage zones using connected-component labeling
and contour extraction, attributes individual/compound damage modes, computes
normalized severity scores, and generates explainable diagnostic rationales.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

try:
    from .ml_classifier import HeritageDamageClassifier, extract_region_features
    _ML_CLASSIFIER_INSTANCE = HeritageDamageClassifier()
except Exception:
    _ML_CLASSIFIER_INSTANCE = None


DAMAGE_TYPE_NAMES = {
    "crack": "Structural / Hairline Crack",
    "moisture": "Moisture Stains & Seepage",
    "peeling": "Peeling Paint / Delaminated Plaster",
    "biological": "Biological Growth (Moss/Algae)",
    "erosion": "Surface Erosion & Weathering",
    "compound": "Compound Multi-Hazard Damage"
}


def analyze_damage_regions(
    composite_mask: np.ndarray,
    crack_mask: np.ndarray,
    moisture_mask: np.ndarray,
    peeling_mask: np.ndarray,
    biological_mask: np.ndarray,
    erosion_mask: np.ndarray,
    fused_density: np.ndarray,
    min_region_area: int = 150,
    rgb_image: Optional[np.ndarray] = None,
    gray_image: Optional[np.ndarray] = None,
    hsv_image: Optional[np.ndarray] = None,
    lab_image: Optional[np.ndarray] = None
) -> List[Dict[str, Any]]:
    """
    Extract, localize, and attribute individual deterioration regions.

    Args:
        composite_mask: Binary uint8 composite damage mask.
        crack_mask: Binary uint8 crack mask.
        moisture_mask: Binary uint8 moisture mask.
        peeling_mask: Binary uint8 peeling mask.
        biological_mask: Binary uint8 biological mask.
        erosion_mask: Binary uint8 erosion mask.
        fused_density: Float32 fused density map.
        min_region_area: Minimum pixel area to retain as a significant region.

    Returns:
        List of dicts, each detailing:
            - 'region_id': str (e.g., 'REG-01')
            - 'bbox': tuple (x, y, w, h)
            - 'centroid': tuple (cx, cy)
            - 'area_px': int area in pixels
            - 'area_pct': float percentage of wall area
            - 'aspect_ratio': float elongation ratio
            - 'contour': np.ndarray external contour points
            - 'primary_damage': str canonical damage key
            - 'primary_damage_name': str human-readable name
            - 'channel_overlaps': dict of overlap percentages
            - 'is_compound': bool whether multiple distinct damages overlap
            - 'mean_density': float mean fused density inside region
            - 'severity_score': float score in [0, 100]
            - 'diagnostic_explanation': str detailed explainable reason
    """
    h, w = composite_mask.shape[:2]
    total_pixels = h * w

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(composite_mask, connectivity=8)

    regions_list: List[Dict[str, Any]] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_region_area:
            continue

        comp_x = int(stats[label, cv2.CC_STAT_LEFT])
        comp_y = int(stats[label, cv2.CC_STAT_TOP])
        comp_w = int(stats[label, cv2.CC_STAT_WIDTH])
        comp_h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = int(centroids[label][0]), int(centroids[label][1])

        comp_mask = (labels == label).astype(np.uint8) * 255

        # Extract external contour
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # Aspect ratio
        major_side = max(comp_w, comp_h)
        minor_side = max(1, min(comp_w, comp_h))
        aspect_ratio = round(major_side / float(minor_side), 2)

        # Area metrics
        area_pct = round((area / float(total_pixels)) * 100.0, 3)

        # Measure overlap with each defect channel
        c_crack = cv2.countNonZero(cv2.bitwise_and(comp_mask, crack_mask))
        c_moist = cv2.countNonZero(cv2.bitwise_and(comp_mask, moisture_mask))
        c_peel = cv2.countNonZero(cv2.bitwise_and(comp_mask, peeling_mask))
        c_bio = cv2.countNonZero(cv2.bitwise_and(comp_mask, biological_mask))
        c_eros = cv2.countNonZero(cv2.bitwise_and(comp_mask, erosion_mask))

        counts = {
            "crack": c_crack,
            "moisture": c_moist,
            "peeling": c_peel,
            "biological": c_bio,
            "erosion": c_eros
        }

        # Percentage overlaps relative to region area
        overlaps = {k: round((v / float(area)) * 100.0, 1) for k, v in counts.items()}

        # Identify primary and secondary damages
        sorted_channels = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
        top_channel, top_pct = sorted_channels[0]
        second_channel, second_pct = sorted_channels[1]

        # Compound damage threshold: second damage type covers >= 20% of region
        is_compound = (top_pct >= 25.0 and second_pct >= 20.0)

        if is_compound:
            primary_key = "compound"
            primary_name = f"Compound ({DAMAGE_TYPE_NAMES[top_channel].split('/')[0].strip()} + {DAMAGE_TYPE_NAMES[second_channel].split('/')[0].strip()})"
        elif top_pct > 0:
            primary_key = top_channel
            primary_name = DAMAGE_TYPE_NAMES[top_channel]
        else:
            # Fallback to general erosion / surface anomaly
            primary_key = "erosion"
            primary_name = DAMAGE_TYPE_NAMES["erosion"]

        # Mean fused density inside this region
        mean_density = float(cv2.mean(fused_density, mask=comp_mask)[0])

        # Compute Normalized Severity Score [0 - 100]
        # Base weight based on danger to monument:
        base_weights = {
            "crack": 1.25,        # High structural hazard
            "moisture": 1.20,     # High degradation catalyst
            "compound": 1.30,     # Synergistic multi-hazard
            "peeling": 1.05,      # High cultural/fresco loss
            "biological": 0.95,   # Medium biochemical risk
            "erosion": 0.90       # Medium slow weathering
        }
        b_weight = base_weights.get(primary_key, 1.0)

        # Scale by density and area
        # A region covering 3% of wall is very large
        area_factor = min(35.0, (area_pct / 3.0) * 35.0)
        density_factor = min(45.0, (mean_density / 0.6) * 45.0)
        aspect_bonus = min(20.0, (aspect_ratio / 5.0) * 20.0) if primary_key in ("crack", "compound") else 0.0

        raw_sev = (area_factor + density_factor + aspect_bonus) * b_weight
        severity_score = float(np.clip(raw_sev, 10.0, 100.0))

        # Generate Explainable Diagnostic Rationale
        if primary_key == "crack":
            explanation = (
                f"Crack fracture with {top_pct}% regional dominance (aspect ratio {aspect_ratio}). "
                f"Thin continuous discontinuity (~{comp_w}x{comp_h}px span) exhibiting sharp gradient relief, "
                f"threatening structural joint detachment and stress-induced material splitting."
            )
        elif primary_key == "moisture":
            explanation = (
                f"Moisture seepage plume with {top_pct}% regional dominance (covering {area_pct}% of surface). "
                f"Exhibits notable darkness depression and damp tide boundaries indicative of active capillary rise, "
                f"which dissolves binders and promotes efflorescence."
            )
        elif primary_key == "peeling":
            explanation = (
                f"Peeling/flaking plaster with {top_pct}% regional dominance. "
                f"Dense concentration of curling micro-edges (mean density {mean_density:.2f}) and localized "
                f"variance step-change where paint has separated from stone underlayer."
            )
        elif primary_key == "biological":
            explanation = (
                f"Biological biofilm with {top_pct}% regional dominance. "
                f"Spectroscopic chlorophyll signature (Excess Green) and green-yellow chromaticity confirm active "
                f"moss/lichen colonization inducing mechanical rhizoid wedging and biochemical pitting."
            )
        elif primary_key == "erosion":
            explanation = (
                f"Surface erosion with {top_pct}% regional dominance. "
                f"High local variance and texture roughness indicate granular disintegration, "
                f"alveolar pitting, and loss of original carved decorative relief."
            )
        else:  # Compound
            explanation = (
                f"Compound deterioration combining {top_channel.capitalize()} ({top_pct}%) and "
                f"{second_channel.capitalize()} ({second_pct}%). "
                f"Synergistic multi-hazard decay where moisture retention or structural cracking interacts "
                f"with biological growth/flaking, accelerating substrate failure."
            )

        # ML Model Inference (if trained model and images available)
        ml_result = None
        if (
            _ML_CLASSIFIER_INSTANCE is not None and
            _ML_CLASSIFIER_INSTANCE.is_loaded and
            rgb_image is not None and
            gray_image is not None and
            hsv_image is not None and
            lab_image is not None
        ):
            feat_vec = extract_region_features(
                rgb_image=rgb_image,
                gray_image=gray_image,
                hsv_image=hsv_image,
                lab_image=lab_image,
                comp_mask=comp_mask,
                bbox=(comp_x, comp_y, comp_w, comp_h)
            )
            ml_result = _ML_CLASSIFIER_INSTANCE.predict(
                feat_vec,
                fallback_rule={"primary_damage": primary_key}
            )

        regions_list.append({
            "region_id": "",  # Assigned after sorting
            "bbox": (comp_x, comp_y, comp_w, comp_h),
            "centroid": (cx, cy),
            "area_px": area,
            "area_pct": area_pct,
            "aspect_ratio": aspect_ratio,
            "contour": cnt,
            "primary_damage": primary_key,
            "primary_damage_name": primary_name,
            "channel_overlaps": overlaps,
            "is_compound": is_compound,
            "mean_density": round(mean_density, 3),
            "severity_score": round(severity_score, 1),
            "diagnostic_explanation": explanation,
            "ml_inference": ml_result
        })

    # Sort descending by severity score
    regions_list.sort(key=lambda x: x["severity_score"], reverse=True)

    # Assign sequential IDs
    for idx, reg in enumerate(regions_list):
        reg["region_id"] = f"REG-{idx + 1:02d}"

    return regions_list
