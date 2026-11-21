"""
KalaRaksha - Visual Overlays & Explainable Chart Generation Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module handles:
1. Drawing color-coded polygonal contour fills, bounding boxes, and badge labels
2. Strict heritage priority color-coding:
   - Red (#E53935): Critical
   - Orange (#FB8C00): High
   - Yellow (#FDD835): Moderate
   - Green (#43A047): Low
3. Generating analytical figures (damage breakdown donut chart, priority distribution)
4. Side-by-side comparative views and colored channel masks
"""

from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io


PRIORITY_COLORS_RGB = {
    "CRITICAL": (229, 57, 53),     # Red
    "HIGH": (251, 140, 0),         # Orange
    "MODERATE": (253, 216, 53),    # Yellow
    "LOW": (67, 160, 71)           # Green
}


def draw_damage_overlay(
    base_rgb: np.ndarray,
    regions: List[Dict[str, Any]],
    draw_contours: bool = True,
    draw_boxes: bool = True,
    draw_labels: bool = True,
    fill_alpha: float = 0.35,
    font_scale: float = 0.55
) -> np.ndarray:
    """
    Draw high-clarity explainable annotations on the base heritage wall image.

    Args:
        base_rgb: RGB uint8 image.
        regions: List of annotated region dictionaries.
        draw_contours: Whether to fill and outline the exact defect contour.
        draw_boxes: Whether to draw bounding rectangles.
        draw_labels: Whether to render region ID, defect type, and severity badges.
        fill_alpha: Transparency of contour fill.
        font_scale: Size of badge text.

    Returns:
        Annotated RGB uint8 image.
    """
    h, w = base_rgb.shape[:2]
    annotated = base_rgb.copy()
    overlay_fill = base_rgb.copy()

    # 1. First pass: Draw all semi-transparent contour fills
    if draw_contours:
        for reg in regions:
            priority = reg.get("priority", "MODERATE")
            color = PRIORITY_COLORS_RGB.get(priority, (253, 216, 53))
            cnt = reg.get("contour")
            if cnt is not None and len(cnt) > 0:
                cv2.drawContours(overlay_fill, [cnt], -1, color, cv2.FILLED)

        # Blend fills with base
        cv2.addWeighted(overlay_fill, fill_alpha, annotated, 1.0 - fill_alpha, 0, annotated)

    # 2. Second pass: Draw crisp boundaries, bounding boxes, and badge labels
    for reg in regions:
        priority = reg.get("priority", "MODERATE")
        color = PRIORITY_COLORS_RGB.get(priority, (253, 216, 53))
        reg_id = reg.get("region_id", "REG")
        p_name = reg.get("primary_damage", "DETERIORATION").upper()
        sev = reg.get("severity_score", 0.0)

        # Draw contour perimeter line
        cnt = reg.get("contour")
        if draw_contours and cnt is not None and len(cnt) > 0:
            cv2.drawContours(annotated, [cnt], -1, color, 2, cv2.LINE_AA)

        # Draw bounding box
        bx, by, bw, bh = reg.get("bbox", (0, 0, 0, 0))
        if draw_boxes and bw > 0 and bh > 0:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), color, 2, cv2.LINE_AA)

        # Draw Badge Header
        if draw_labels and bw > 0:
            badge_text = f"{reg_id} [{p_name}] {sev:.0f}%"

            # Determine text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

            # Position badge above bbox, or inside if near top edge
            badge_y2 = by - 4 if (by - text_h - 10) > 0 else (by + text_h + 10)
            badge_y1 = badge_y2 - text_h - 8
            badge_x1 = max(2, bx)
            badge_x2 = min(w - 2, badge_x1 + text_w + 12)

            # Draw badge background (solid priority color or dark rectangle)
            cv2.rectangle(
                annotated,
                (badge_x1, badge_y1),
                (badge_x2, badge_y2),
                color,
                cv2.FILLED
            )

            # Text color: black for yellow/light colors, white for dark red/green/orange
            if priority in ("MODERATE", "LOW"):
                text_color = (20, 20, 20)
            else:
                text_color = (255, 255, 255)

            text_pos_x = badge_x1 + 6
            text_pos_y = badge_y2 - 5
            cv2.putText(
                annotated,
                badge_text,
                (text_pos_x, text_pos_y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA
            )

    return annotated


def create_damage_analytics_plot(
    channel_contributions: Dict[str, float],
    priority_counts: Dict[str, int],
    health_index: float
) -> plt.Figure:
    """
    Generate an explainable multi-panel Matplotlib figure illustrating:
    1. Deterioration Channel Contribution Donut Chart
    2. Restoration Priority Breakdown Bar Chart
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#FAFAFA")

    # 1. Donut Chart for Deterioration Types
    labels = []
    sizes = []
    colors = []
    channel_color_map = {
        "cracks": "#D32F2F",       # Deep Red
        "moisture": "#1976D2",     # Blue
        "peeling": "#F57C00",      # Deep Orange
        "biological": "#388E3C",   # Forest Green
        "erosion": "#795548"       # Earth Brown
    }

    for ch, pct in channel_contributions.items():
        if pct > 0:
            labels.append(ch.capitalize())
            sizes.append(pct)
            colors.append(channel_color_map.get(ch, "#9E9E9E"))

    if not sizes:
        sizes = [100.0]
        labels = ["No Defects"]
        colors = ["#4CAF50"]

    wedges, texts, autotexts = ax1.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.75,
        textprops=dict(color="#212121", fontsize=9, weight="bold")
    )

    # Donut center circle
    centre_circle = plt.Circle((0, 0), 0.55, fc="#FAFAFA")
    ax1.add_artist(centre_circle)
    ax1.text(
        0, 0,
        f"HHI\n{health_index:.0f}/100",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#37474F"
    )
    ax1.set_title("Deterioration Mechanism Share", fontsize=11, fontweight="bold", pad=10)

    # 2. Priority Level Bar Chart
    priorities = ["CRITICAL", "HIGH", "MODERATE", "LOW"]
    bar_counts = [priority_counts.get(p, 0) for p in priorities]
    bar_colors = ["#E53935", "#FB8C00", "#FDD835", "#43A047"]

    y_pos = np.arange(len(priorities))
    bars = ax2.barh(y_pos, bar_counts, color=bar_colors, edgecolor="#424242", height=0.55)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(priorities, fontsize=10, fontweight="bold")
    ax2.invert_yaxis()  # Critical at top
    ax2.set_xlabel("Number of Detected Regions", fontsize=10, fontweight="bold")
    ax2.set_title("Restoration Priority Distribution", fontsize=11, fontweight="bold", pad=10)
    ax2.grid(axis="x", linestyle="--", alpha=0.5)

    # Add numeric labels on bars
    for bar in bars:
        width = bar.get_width()
        ax2.text(
            width + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#212121"
        )

    plt.tight_layout()
    return fig


def create_channel_thumbnail(
    base_rgb: np.ndarray,
    binary_mask: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Highlight a single binary mask over the heritage wall background.
    """
    out = base_rgb.copy()
    if cv2.countNonZero(binary_mask) == 0:
        return out

    color_layer = np.full_like(base_rgb, color_rgb, dtype=np.uint8)
    mask_bool = binary_mask > 0

    blended = (out * (1.0 - alpha) + color_layer * alpha).astype(np.uint8)
    out[mask_bool] = blended[mask_bool]

    # Draw sharp contour
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_rgb, 2, cv2.LINE_AA)
    return out
