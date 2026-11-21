"""
KalaRaksha - PDF Conservation Report & CSV Export Module
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module creates:
1. Multi-page, professional PDF Conservation Dossiers using ReportLab:
   - Executive conservation summary
   - 4-plate photographic evidence sheet (Original, CLAHE-Enhanced, Heatmap, Overlay)
   - Comprehensive damage inventory table
   - International charter-aligned conservation protocols (ICOMOS / ICCROM / ASI)
   - Conservator attestation block
2. Machine-readable CSV inventory export with pandas
"""

from typing import Dict, Any, List, Optional
import io
import html
import datetime
import numpy as np
import pandas as pd
from PIL import Image

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak, HRFlowable
)


def generate_csv_report(
    regions: List[Dict[str, Any]],
    overall_metrics: Dict[str, Any]
) -> bytes:
    """
    Generate downloadable CSV byte stream containing full damage zone inventory.
    """
    rows = []
    for reg in regions:
        recs = reg.get("recommendations", [])
        rec_str = " | ".join(recs) if recs else "Routine monitoring."

        rows.append({
            "Region ID": reg.get("region_id", ""),
            "Primary Damage Type": reg.get("primary_damage_name", ""),
            "Canonical Key": reg.get("primary_damage", ""),
            "Is Compound": "Yes" if reg.get("is_compound", False) else "No",
            "Area (px)": reg.get("area_px", 0),
            "Area (%)": reg.get("area_pct", 0.0),
            "Aspect Ratio": reg.get("aspect_ratio", 1.0),
            "Mean Fused Density": reg.get("mean_density", 0.0),
            "Severity Score (0-100)": reg.get("severity_score", 0.0),
            "Restoration Priority": reg.get("priority", "MODERATE"),
            "Action Urgency": reg.get("urgency_days", ""),
            "Diagnostic Rationale": reg.get("diagnostic_explanation", ""),
            "Conservation Directives": rec_str
        })

    df = pd.DataFrame(rows)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    return csv_buffer.getvalue().encode("utf-8")


def _image_array_to_reportlab_image(img_rgb: np.ndarray, max_w: float = 3.2 * inch, max_h: float = 2.4 * inch) -> RLImage:
    """Convert numpy RGB array to in-memory ReportLab Flowable Image."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    # Compute scaled dimensions
    w, h = pil_img.size
    aspect = float(w) / float(h)

    target_w = max_w
    target_h = target_w / aspect

    if target_h > max_h:
        target_h = max_h
        target_w = target_h * aspect

    return RLImage(buf, width=target_w, height=target_h)


def generate_pdf_report(
    original_rgb: np.ndarray,
    enhanced_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    regions: List[Dict[str, Any]],
    overall_metrics: Dict[str, Any],
    channel_contributions: Dict[str, float],
    image_filename: str = "heritage_monument.jpg"
) -> bytes:
    """
    Generate an authoritative, publication-quality PDF Conservation Dossier.

    Args:
        original_rgb: Unenhanced RGB image.
        enhanced_rgb: CLAHE-enhanced RGB image.
        heatmap_rgb: Fused thermal heatmap RGB image.
        overlay_rgb: Priority-annotated bounding box overlay RGB image.
        regions: List of annotated region dictionaries.
        overall_metrics: Wall-level health index and summary dictionary.
        channel_contributions: Defect percentage share dictionary.
        image_filename: Source photograph name.

    Returns:
        bytes: Raw PDF file byte stream.
    """
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "KalaRakshaTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#7B1113"),  # Deep Terracotta
        alignment=1,  # Center
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        "KalaRakshaSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#424242"),
        alignment=1,
        spaceAfter=12
    )

    section_heading = ParagraphStyle(
        "KalaRakshaSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#7B1113"),
        spaceBefore=10,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "KalaRakshaBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#212121")
    )

    body_bold = ParagraphStyle(
        "KalaRakshaBodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    table_cell = ParagraphStyle(
        "KalaRakshaTableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#212121")
    )

    table_cell_bold = ParagraphStyle(
        "KalaRakshaTableCellBold",
        parent=table_cell,
        fontName="Helvetica-Bold"
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("KALARAKSHA: HERITAGE WALL DETERIORATION DOSSIER", title_style))
    story.append(Paragraph("Image Processing Analysis (IPA) & Restoration Priority Condition Assessment", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#7B1113"), spaceAfter=8))

    # 2. Metadata & Inspection Info Table
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hhi = overall_metrics.get("health_index", 100.0)
    urgency = overall_metrics.get("status_phrase", "Normal")
    priority_level = overall_metrics.get("overall_priority", "LOW")

    meta_data = [
        [
            Paragraph("<b>Target Asset:</b> " + html.escape(str(image_filename)), body_style),
            Paragraph("<b>Inspection Date:</b> " + now_str, body_style)
        ],
        [
            Paragraph("<b>Resolution:</b> " + f"{enhanced_rgb.shape[1]}x{enhanced_rgb.shape[0]} px", body_style),
            Paragraph("<b>Diagnostic Engine:</b> Classical IPA Multi-Modal Fusion", body_style)
        ],
        [
            Paragraph(f"<b>Overall Heritage Health:</b> <font color='#7B1113'><b>{hhi:.1f} / 100</b></font>", body_style),
            Paragraph(f"<b>Restoration Priority:</b> <b>{priority_level}</b> ({urgency})", body_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[260, 260])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F9F6F0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7CCC8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # 3. Defect Share Summary
    contrib_str = " | ".join([f"<b>{k.capitalize()}:</b> {v}%" for k, v in channel_contributions.items() if v > 0])
    if not contrib_str:
        contrib_str = "No active surface defects detected."
    story.append(Paragraph(f"<b>Deterioration Mechanism Share:</b> {contrib_str}", body_style))
    story.append(Spacer(1, 10))

    # 4. Photographic Evidence Plate (2x2 Grid)
    story.append(Paragraph("1. Multi-Modal Photographic Evidence Plates", section_heading))

    img_orig_rl = _image_array_to_reportlab_image(original_rgb, max_w=255, max_h=165)
    img_enh_rl = _image_array_to_reportlab_image(enhanced_rgb, max_w=255, max_h=165)
    img_heat_rl = _image_array_to_reportlab_image(heatmap_rgb, max_w=255, max_h=165)
    img_over_rl = _image_array_to_reportlab_image(overlay_rgb, max_w=255, max_h=165)

    plate_data = [
        [
            Paragraph("<b>Plate A:</b> As-Found Photographic State", body_style),
            Paragraph("<b>Plate B:</b> CLAHE Contrast-Enhanced State", body_style)
        ],
        [img_orig_rl, img_enh_rl],
        [
            Paragraph("<b>Plate C:</b> Fused Thermal Deterioration Heatmap", body_style),
            Paragraph("<b>Plate D:</b> Priority-Tagged Damage Overlay", body_style)
        ],
        [img_heat_rl, img_over_rl]
    ]

    plate_table = Table(plate_data, colWidths=[260, 260])
    plate_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(plate_table)
    story.append(Spacer(1, 10))

    # 5. Damage Inventory Table (Up to top 15 regions)
    story.append(Paragraph("2. Quantitative Deterioration Inventory & Explainable Findings", section_heading))

    inv_headers = [
        Paragraph("<b>ID</b>", table_cell_bold),
        Paragraph("<b>Damage Type</b>", table_cell_bold),
        Paragraph("<b>Area (%)</b>", table_cell_bold),
        Paragraph("<b>Sev.</b>", table_cell_bold),
        Paragraph("<b>Priority</b>", table_cell_bold),
        Paragraph("<b>Explainable Diagnostic Rationale</b>", table_cell_bold)
    ]
    inv_rows = [inv_headers]

    top_regions = regions[:14] if regions else []

    if not top_regions:
        inv_rows.append([
            Paragraph("N/A", table_cell),
            Paragraph("Intact Fabric", table_cell),
            Paragraph("0.0%", table_cell),
            Paragraph("0", table_cell),
            Paragraph("LOW", table_cell),
            Paragraph("No discrete deterioration zones exceed detection threshold.", table_cell)
        ])
    else:
        for reg in top_regions:
            pri = reg.get("priority", "MODERATE")
            pri_color = "#E53935" if pri == "CRITICAL" else ("#FB8C00" if pri == "HIGH" else ("#FBC02D" if pri == "MODERATE" else "#43A047"))

            inv_rows.append([
                Paragraph(f"<b>{reg.get('region_id')}</b>", table_cell),
                Paragraph(reg.get("primary_damage_name", ""), table_cell),
                Paragraph(f"{reg.get('area_pct', 0.0):.2f}%", table_cell),
                Paragraph(f"{reg.get('severity_score', 0.0):.0f}", table_cell),
                Paragraph(f"<font color='{pri_color}'><b>{pri}</b></font>", table_cell),
                Paragraph(reg.get("diagnostic_explanation", ""), table_cell)
            ])

    inv_table = Table(inv_rows, colWidths=[40, 110, 45, 30, 55, 240])
    inv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7B1113")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#BDBDBD")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FBFBFB")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(inv_table)
    story.append(Spacer(1, 10))

    # 6. Conservation Directives Section
    story.append(Paragraph("3. Actionable Conservation & Restoration Directives", section_heading))
    story.append(Paragraph(
        "Interventions formulated in strict accordance with the ICOMOS Venice Charter, "
        "ICCROM Stone Conservation Guidelines, and the Archaeological Survey of India (ASI) standards.",
        body_style
    ))
    story.append(Spacer(1, 4))

    # Gather unique recommendations per priority tier
    crit_recs = []
    high_recs = []
    mod_recs = []

    for reg in regions:
        pri = reg.get("priority", "MODERATE")
        recs = reg.get("recommendations", [])
        if pri == "CRITICAL":
            for r in recs:
                if r not in crit_recs:
                    crit_recs.append(r)
        elif pri == "HIGH":
            for r in recs:
                if r not in high_recs:
                    high_recs.append(r)
        else:
            for r in recs:
                if r not in mod_recs:
                    mod_recs.append(r)

    if crit_recs:
        story.append(Paragraph("<b><font color='#E53935'>[CRITICAL PRIORITY - IMMEDIATE INTERVENTION (72 HOURS)]</font></b>", body_style))
        for r in crit_recs[:4]:
            story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 4))

    if high_recs:
        story.append(Paragraph("<b><font color='#FB8C00'>[HIGH PRIORITY - CONSERVATION PROGRAM (30-60 DAYS)]</font></b>", body_style))
        for r in high_recs[:4]:
            story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 4))

    if mod_recs or (not crit_recs and not high_recs):
        story.append(Paragraph("<b><font color='#388E3C'>[PREVENTIVE MAINTENANCE & PERIODIC MONITORING]</font></b>", body_style))
        sample_mod = mod_recs[:3] if mod_recs else [
            "Maintain seasonal photographic monitoring of historic surfaces.",
            "Inspect perimeter drainage gutters and downspouts before rainy seasons.",
            "Avoid aggressive synthetic pressure washing or unauthorized chemical cleaning."
        ]
        for r in sample_mod:
            story.append(Paragraph(f"• {r}", body_style))
        story.append(Spacer(1, 8))

    # 7. Conservator Sign-Off Block
    story.append(Spacer(1, 6))
    sign_data = [
        [
            Paragraph("<b>Lead Heritage Conservator:</b> _______________________", body_style),
            Paragraph("<b>Superintending Archaeologist:</b> _______________________", body_style)
        ],
        [
            Paragraph("<b>Affiliation / Agency:</b> ___________________________", body_style),
            Paragraph("<b>Field Inspection Stamp:</b> [  VERIFIED AUTHENTIC  ]", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[260, 260])
    sign_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(sign_table)

    # Build PDF document
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
