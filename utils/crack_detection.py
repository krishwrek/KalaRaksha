"""
KalaRaksha - Crack Detection Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module implements classical Image Processing Analysis (IPA) to detect,
filter, and quantify structural and hairline cracks on heritage surfaces.

Methods used:
1. Morphological Black-Hat transform to isolate dark linear fissures
2. Adaptive Canny edge detection with dynamic thresholds
3. Morphological bridge & closing operations
4. Probabilistic Hough Line Transform (HoughLinesP)
5. Geometric filtering (elongation, tortuosity, thickness, circularity)
6. Morphological skeletonization for precision length & branching quantification
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import cv2


def _morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Compute 1-pixel wide skeleton using iterative morphological thinning
    (Guo-Hall / standard morphological skeletonization with cv2).
    """
    img = binary_mask.copy()
    skeleton = np.zeros(img.shape, dtype=np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    max_iter = 100
    iteration = 0
    while iteration < max_iter:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break
        iteration += 1

    return skeleton


def detect_cracks(
    gray_image: np.ndarray,
    sensitivity: float = 0.5,
    min_length: int = 25,
    max_thickness: int = 18
) -> Dict[str, Any]:
    """
    Detect cracks in heritage wall grayscale image using classical IPA.

    Args:
        gray_image: Grayscale uint8 image.
        sensitivity: Float in [0.1, 1.0]. Higher values detect subtle hairline cracks.
        min_length: Minimum crack length in pixels to filter out isolated micro-noise.
        max_thickness: Maximum estimated thickness for a fissure to prevent flagging wide decorative reliefs.

    Returns:
        dict containing:
            - 'crack_mask': uint8 binary mask (0 or 255)
            - 'crack_density': float32 normalized map [0.0, 1.0]
            - 'skeleton_mask': uint8 1-pixel centerline skeleton
            - 'crack_lines': list of (x1, y1, x2, y2) line segments
            - 'total_crack_length_px': int total length of all crack skeletons
            - 'max_crack_length_px': int longest continuous crack segment
            - 'num_crack_segments': int count of validated crack branches
            - 'mean_thickness_px': float estimated mean crack width
            - 'coverage_pct': float percentage of wall affected by cracks
            - 'severity_score': float score in [0, 100]
            - 'explanation': str detailed explainable diagnostic rationale
    """
    h, w = gray_image.shape
    total_pixels = h * w

    # Sensitivity normalization (maps [0.1, 1.0] -> internal scaling)
    sens = np.clip(sensitivity, 0.1, 1.0)

    # 1. Morphological Black-Hat transform: highlights dark fissures on lighter masonry
    # We combine two scales: fine (7x7) and medium (15x15) to capture both hairline and wide structural fractures
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    blackhat_s = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel_small)
    blackhat_m = cv2.morphologyEx(gray_image, cv2.MORPH_BLACKHAT, kernel_med)
    blackhat = cv2.addWeighted(blackhat_s, 0.5, blackhat_m, 0.5, 0)

    # 2. Adaptive Canny Edge Detection
    # Calculate median intensity to anchor dynamic thresholds
    med_val = np.median(gray_image)
    sigma = 0.35 * (1.6 - 0.7 * sens)
    lower_thresh = int(max(15, (1.0 - sigma) * med_val))
    upper_thresh = int(min(240, (1.0 + sigma) * med_val))

    canny_raw = cv2.Canny(gray_image, lower_thresh, upper_thresh)

    # 3. Fuse Black-Hat features with Canny edges
    # Black-hat thresholded with Otsu or sensitivity cutoff
    bh_thresh_val = int(max(12, 45 - sens * 30))
    _, bh_binary = cv2.threshold(blackhat, bh_thresh_val, 255, cv2.THRESH_BINARY)

    # Combined candidate edges
    candidate_mask = cv2.bitwise_or(canny_raw, bh_binary)

    # 4. Morphological bridge & closing: connect micro-fracture gaps
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_closed = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, close_kernel)

    # 5. Probabilistic Hough Line Transform to capture continuous linear propagation
    hough_thresh = int(max(20, 55 - sens * 35))
    min_line_len = int(max(15, min_length * (1.2 - 0.4 * sens)))
    max_line_gap = int(6 + sens * 8)

    lines = cv2.HoughLinesP(
        candidate_closed,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )

    hough_mask = np.zeros((h, w), dtype=np.uint8)
    detected_lines: List[Tuple[int, int, int, int]] = []
    if lines is not None:
        for line in lines:
            flat_line = line.reshape(-1)
            if len(flat_line) >= 4:
                x1, y1, x2, y2 = int(flat_line[0]), int(flat_line[1]), int(flat_line[2]), int(flat_line[3])
                detected_lines.append((x1, y1, x2, y2))
                cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)

    # 6. Geometric Feature Filtering (Length, Thickness, Circularity, Elongation)
    # Distinguish cracks from regular stone edges, mortar joints, and decorative motifs
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate_closed, connectivity=8)

    validated_crack_mask = np.zeros((h, w), dtype=np.uint8)
    segment_lengths: List[int] = []
    segment_thicknesses: List[float] = []

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        comp_w = stats[label, cv2.CC_STAT_WIDTH]
        comp_h = stats[label, cv2.CC_STAT_HEIGHT]

        if area < 10:  # Skip micro-noise
            continue

        # Elongation / Aspect Ratio
        major_side = max(comp_w, comp_h)
        minor_side = max(1, min(comp_w, comp_h))
        aspect_ratio = major_side / float(minor_side)

        # Extract single component mask
        comp_mask = (labels == label).astype(np.uint8) * 255

        # Compute perimeter and circularity
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = contours[0]
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        # Circularity = 4 * pi * Area / (Perimeter^2)
        # Cracks are highly tortuous and non-circular (circularity << 0.2)
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)

        # Estimated average thickness
        # For a thin strip, Area ~ Length * Thickness, Perimeter ~ 2 * Length
        estimated_length = perimeter / 2.0
        estimated_thickness = area / max(1.0, estimated_length)

        # Check overlap with Hough linear structures
        hough_overlap = cv2.countNonZero(cv2.bitwise_and(comp_mask, hough_mask))
        has_linear_support = hough_overlap > 0 or aspect_ratio > 3.0

        # Strict IPA filtering rule:
        # A valid crack must be elongated (aspect_ratio > 2.5 or linear support),
        # low circularity (< 0.35), thin (thickness <= max_thickness), and exceed minimum length.
        is_crack = (
            (aspect_ratio >= 2.5 or has_linear_support)
            and circularity <= 0.35
            and estimated_thickness <= max_thickness
            and (major_side >= min_length or hough_overlap > 10)
        )

        if is_crack:
            validated_crack_mask = cv2.bitwise_or(validated_crack_mask, comp_mask)
            segment_lengths.append(int(major_side))
            segment_thicknesses.append(float(estimated_thickness))

    # Also blend in supported Hough lines that align with blackhat features
    hough_on_blackhat = cv2.bitwise_and(hough_mask, bh_binary)
    if cv2.countNonZero(hough_on_blackhat) > 0:
        validated_crack_mask = cv2.bitwise_or(validated_crack_mask, hough_on_blackhat)

    # 7. Skeletonization for accurate centerline length & branching
    skeleton = _morphological_skeleton(validated_crack_mask)
    total_crack_len = int(cv2.countNonZero(skeleton))

    # Calculate statistics
    num_segments = len(segment_lengths)
    max_crack_len = int(max(segment_lengths)) if segment_lengths else 0
    mean_thickness = float(np.mean(segment_thicknesses)) if segment_thicknesses else 0.0

    crack_pixel_count = cv2.countNonZero(validated_crack_mask)
    coverage_pct = round((crack_pixel_count / float(total_pixels)) * 100.0, 3)

    # 8. Normalized Density Map (smooth Gaussian representation for heatmap fusion)
    crack_density = cv2.GaussianBlur(validated_crack_mask.astype(np.float32) / 255.0, (15, 15), 0)
    max_d = np.max(crack_density)
    if max_d > 0:
        crack_density = crack_density / max_d

    # 9. Severity Score Calculation [0 - 100]
    # Factors: max crack length, total crack length relative to image dimension, thickness
    diag_length = np.sqrt(h**2 + w**2)
    length_ratio = total_crack_len / max(1.0, diag_length)
    max_len_ratio = max_crack_len / max(1.0, diag_length)

    raw_severity = (
        (length_ratio * 45.0) +
        (max_len_ratio * 35.0) +
        (min(1.0, mean_thickness / 8.0) * 20.0)
    )
    severity_score = float(np.clip(raw_severity * 1.5, 0.0, 100.0))

    # 10. Explainable diagnostic rationale
    if num_segments == 0:
        explanation = "No structural or hairline cracks detected. Surface masonry shows continuous tensile cohesion."
    else:
        explanation = (
            f"Detected {num_segments} crack segment(s) totaling ~{total_crack_len}px in length "
            f"(longest fissure: {max_crack_len}px, avg width: {mean_thickness:.1f}px). "
            f"The structures exhibit high elongation (aspect ratio > 2.5), low circularity (< 0.35), "
            f"and linear Hough continuity indicative of tensile or shear stress fracturing."
        )

    return {
        "crack_mask": validated_crack_mask,
        "crack_density": crack_density,
        "skeleton_mask": skeleton,
        "crack_lines": detected_lines,
        "total_crack_length_px": total_crack_len,
        "max_crack_length_px": max_crack_len,
        "num_crack_segments": num_segments,
        "mean_thickness_px": round(mean_thickness, 2),
        "coverage_pct": coverage_pct,
        "severity_score": round(severity_score, 1),
        "explanation": explanation
    }
