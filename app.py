"""
KalaRaksha: IPA-Based Heritage-Wall Deterioration Analysis & Restoration-Priority Mapping System

A classical Image Processing Analysis (IPA) system engineered to detect:
1. Structural & Hairline Cracks (Black-Hat morphology, adaptive Canny, Hough transform)
2. Moisture Stains & Rising Damp (HSV/LAB luminance depression & amoeboid contours)
3. Peeling Paint & Flaking Plaster (Sobel gradient magnitude & local edge density)
4. Biological Growth & Lichen (Excess Green Index & dual-band chlorophyll segmentation)
5. Surface Erosion & Weathering (Local variance, Shannon entropy, and Laplacian roughness)

Features:
- Explainable visual heatmaps & color-coded bounding box overlays
- Heritage Health Index (HHI, 0-100) and restoration priority classification
- Tailored conservation directives aligned with ICOMOS, ICCROM, and ASI standards
- One-click PDF Conservation Dossier & CSV Inventory generation
"""

import os
import io
import datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2

from dotenv import load_dotenv
load_dotenv()

# Import KalaRaksha IPA Modules
from utils.preprocessing import preprocess_heritage_image
from utils.crack_detection import detect_cracks
from utils.moisture_detection import detect_moisture_seepage
from utils.peeling_detection import detect_peeling_plaster
from utils.biological_growth import detect_biological_growth
from utils.erosion_detection import detect_surface_erosion
from utils.heatmap_fusion import fuse_deterioration_channels, COLORMAP_REGISTRY
from utils.region_analysis import analyze_damage_regions
from utils.priority_engine import (
    classify_region_priority,
    compute_heritage_health_index,
    PRIORITY_METADATA
)
from utils.visualization import (
    draw_damage_overlay,
    create_damage_analytics_plot,
    create_channel_thumbnail
)
from utils.report_generator import generate_csv_report, generate_pdf_report
from utils.security import validate_image_payload, safe_html


# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="KalaRaksha | Heritage Wall Deterioration Analysis",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Heritage Theme CSS
st.markdown("""
<style>
    /* Global Heritage Palette */
    :root {
        --heritage-terracotta: #7B1113;
        --heritage-gold: #C5A059;
        --heritage-sand: #F7F4EF;
        --heritage-slate: #2B2D42;
        --priority-critical: #E53935;
        --priority-high: #FB8C00;
        --priority-moderate: #FDD835;
        --priority-low: #43A047;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #4A0E17 0%, #7B1113 50%, #8D1B1D 100%);
        color: #FFFFFF;
        padding: 24px 28px;
        border-radius: 12px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(123, 17, 19, 0.25);
        border-left: 6px solid #C5A059;
    }
    .hero-banner h1 {
        color: #FFFFFF;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .hero-banner p {
        color: #F7E7CE;
        margin: 6px 0 0 0;
        font-size: 1.02rem;
    }

    /* Metric Cards */
    .metric-card {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #E0DCD3;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        text-align: center;
        transition: transform 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #616161;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Priority Badges */
    .badge-critical {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        border: 1px solid #FFCDD2;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        border: 1px solid #FFE0B2;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #FFFDE7;
        color: #F57F17;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        border: 1px solid #FFF9C4;
        display: inline-block;
    }
    .badge-low {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        border: 1px solid #C8E6C9;
        display: inline-block;
    }

    /* Channel Cards in Tab 2 */
    .channel-box {
        background-color: #FAFAF8;
        border: 1px solid #E7E3DA;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# SIDEBAR CONFIGURATION & CONTROLS
# ==========================================
with st.sidebar:
    st.markdown("## 🏛️ **KalaRaksha**")
    st.caption("IPA Heritage Wall Deterioration & Restoration Mapping")
    st.markdown("---")

    # Mode Selection
    input_mode = st.radio(
        "**Image Source:**",
        ["🖼️ Demo Mode (Preset Heritage Walls)", "📤 Upload Custom Wall Photograph"],
        index=0
    )

    image_bytes = None
    image_name = "heritage_wall.jpg"

    SAMPLE_IMAGES = {
        "🏛️ Ancient Granite Temple Wall (Fractures & Fissures)": "sample_images/sample1_temple_crack.jpg",
        "🎨 Historic Cave Mural Fresco (Moisture Seepage)": "sample_images/sample2_mural_moisture.jpg",
        "🏛️ Painted Palace Plaster (Peeling & Delamination)": "sample_images/sample3_fresco_peeling.jpg",
        "🗿 Carved Sandstone Bas-Relief (Moss & Lichen Growth)": "sample_images/sample4_sculpture_moss.jpg",
        "🏛️ Weathered Sandstone Pillar (Granular Pitting & Erosion)": "sample_images/sample5_sandstone_erosion.jpg"
    }

    if input_mode == "🖼️ Demo Mode (Preset Heritage Walls)":
        selected_sample = st.selectbox("Choose Heritage Asset:", list(SAMPLE_IMAGES.keys()))
        sample_path = SAMPLE_IMAGES[selected_sample]
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                image_bytes = f.read()
            image_name = os.path.basename(sample_path)
        else:
            st.error(f"Sample asset not found at {sample_path}. Please run create_sample_assets.py.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Wall Photograph (JPG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            raw_payload = uploaded_file.read()
            is_valid, safe_name, err_msg = validate_image_payload(raw_payload, uploaded_file.name)
            if not is_valid:
                st.error(f"❌ Upload Security Error: {safe_html(err_msg)}")
                image_bytes = None
            else:
                image_bytes = raw_payload
                image_name = safe_name

    st.markdown("---")
    st.markdown("### ⚙️ **IPA Detection Controls**")

    # Global Sensitivity Slider
    sensitivity = st.slider(
        "Detection Sensitivity:",
        min_value=0.10,
        max_value=1.00,
        value=0.55,
        step=0.05,
        help="Calibrates edge thresholds, color tolerance, and texture roughness cutoffs."
    )
    sens_label = "Balanced" if 0.4 <= sensitivity <= 0.7 else ("High (Fine Micro-Defects)" if sensitivity > 0.7 else "Low (Severe Only)")
    st.caption(f"Profile: **{sens_label}**")

    # Feature Toggles
    st.markdown("**Active Deterioration Detectors:**")
    toggle_cracks = st.toggle("⚡ Structural & Hairline Cracks", value=True)
    toggle_moisture = st.toggle("💧 Moisture Stains & Seepage", value=True)
    toggle_peeling = st.toggle("🍂 Peeling Paint & Plaster", value=True)
    toggle_biological = st.toggle("🌿 Biological Growth (Moss/Algae)", value=True)
    toggle_erosion = st.toggle("🌪️ Surface Erosion & Weathering", value=True)

    with st.expander("🛠️ Preprocessing & Heatmap Settings"):
        apply_clahe = st.checkbox("Apply LAB CLAHE Contrast Boost", value=True)
        clahe_clip = st.slider("CLAHE Clip Limit", 1.0, 4.0, 2.5, 0.5)
        filter_type = st.selectbox("Noise Reduction Filter", ["bilateral", "median", "none"], index=0)
        gamma_val = st.slider("Gamma Correction", 0.7, 1.5, 1.0, 0.05)
        colormap_choice = st.selectbox("Heatmap Colormap", list(COLORMAP_REGISTRY.keys()), index=0)
        alpha_blend = st.slider("Heatmap Overlay Opacity", 0.1, 0.9, 0.55, 0.05)
        min_region_area = st.slider("Min Region Size Filter (px)", 50, 400, 150, 25)


# ==========================================
# MAIN APPLICATION BODY
# ==========================================

# 1. Hero Header Banner
st.markdown("""
<div class="hero-banner">
    <h1>🏛️ KalaRaksha: Heritage Conservation Intelligence</h1>
    <p>Classical Image-Processing Analysis (IPA) & Restoration-Priority Mapping for Monuments, Murals, Sculptures & Historic Walls</p>
</div>
""", unsafe_allow_html=True)

if image_bytes is None:
    st.info("👈 Select a sample heritage asset from Demo Mode in the sidebar or upload a photograph to begin.")
    st.stop()


# ==========================================
# EXECUTE CORE IPA PIPELINE
# ==========================================
with st.spinner("Analyzing heritage wall fabric using classical IPA engines..."):
    # Step 1: Preprocessing
    prep_data = preprocess_heritage_image(
        image_bytes,
        max_dim=1280,
        apply_clahe=apply_clahe,
        clahe_clip=clahe_clip,
        gamma=gamma_val,
        filter_type=filter_type
    )

    orig_rgb = prep_data["original_rgb"]
    enh_rgb = prep_data["enhanced_rgb"]
    gray = prep_data["gray"]
    hsv = prep_data["hsv"]
    h_img, w_img = prep_data["dimensions"]

    # Step 2: Individual IPA Detectors
    # Cracks
    if toggle_cracks:
        crack_res = detect_cracks(gray, sensitivity=sensitivity)
    else:
        crack_res = {
            "crack_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "crack_density": np.zeros((h_img, w_img), dtype=np.float32),
            "total_crack_length_px": 0, "max_crack_length_px": 0, "num_crack_segments": 0,
            "mean_thickness_px": 0.0, "coverage_pct": 0.0, "severity_score": 0.0,
            "explanation": "Crack detection channel disabled."
        }

    # Moisture
    if toggle_moisture:
        moist_res = detect_moisture_seepage(enh_rgb, hsv, sensitivity=sensitivity)
    else:
        moist_res = {
            "moisture_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "moisture_density": np.zeros((h_img, w_img), dtype=np.float32),
            "moisture_area_pct": 0.0, "num_patches": 0, "largest_patch_pct": 0.0,
            "severity_score": 0.0, "explanation": "Moisture detection channel disabled."
        }

    # Peeling
    if toggle_peeling:
        peel_res = detect_peeling_plaster(gray, enh_rgb, sensitivity=sensitivity)
    else:
        peel_res = {
            "peeling_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "peeling_density": np.zeros((h_img, w_img), dtype=np.float32),
            "peeling_area_pct": 0.0, "num_peeling_patches": 0, "severity_score": 0.0,
            "explanation": "Peeling plaster detection channel disabled."
        }

    # Biological
    if toggle_biological:
        bio_res = detect_biological_growth(enh_rgb, hsv, sensitivity=sensitivity)
    else:
        bio_res = {
            "biological_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "biological_density": np.zeros((h_img, w_img), dtype=np.float32),
            "coverage_pct": 0.0, "colony_count": 0, "severity_score": 0.0,
            "explanation": "Biological growth detection channel disabled."
        }

    # Erosion
    if toggle_erosion:
        eros_res = detect_surface_erosion(gray, sensitivity=sensitivity)
    else:
        eros_res = {
            "erosion_mask": np.zeros((h_img, w_img), dtype=np.uint8),
            "erosion_roughness": np.zeros((h_img, w_img), dtype=np.float32),
            "eroded_area_pct": 0.0, "num_eroded_patches": 0, "severity_score": 0.0,
            "explanation": "Surface erosion detection channel disabled."
        }

    # Step 3: Weighted Multi-Channel Heatmap Fusion
    toggles_dict = {
        "cracks": toggle_cracks,
        "moisture": toggle_moisture,
        "peeling": toggle_peeling,
        "biological": toggle_biological,
        "erosion": toggle_erosion
    }

    fusion_data = fuse_deterioration_channels(
        crack_density=crack_res["crack_density"],
        moisture_density=moist_res["moisture_density"],
        peeling_density=peel_res["peeling_density"],
        biological_density=bio_res["biological_density"],
        erosion_density=eros_res["erosion_roughness"],
        channel_toggles=toggles_dict,
        enhanced_rgb=enh_rgb,
        colormap_name=colormap_choice,
        alpha_blend=alpha_blend
    )

    # Step 4: Region Localization & Damage Attribution
    regions = analyze_damage_regions(
        composite_mask=fusion_data["composite_mask"],
        crack_mask=crack_res["crack_mask"],
        moisture_mask=moist_res["moisture_mask"],
        peeling_mask=peel_res["peeling_mask"],
        biological_mask=bio_res["biological_mask"],
        erosion_mask=eros_res["erosion_mask"],
        fused_density=fusion_data["fused_density"],
        min_region_area=min_region_area,
        rgb_image=enh_rgb,
        gray_image=gray,
        hsv_image=hsv,
        lab_image=prep_data["lab"]
    )

    # Step 5: Restoration Priority Classification & Heritage Health Index
    for reg in regions:
        pri_meta = classify_region_priority(reg)
        reg.update(pri_meta)

    health_metrics = compute_heritage_health_index(regions, fusion_data["total_damaged_pct"])

    # Step 6: Annotated Visual Overlay
    overlay_rgb = draw_damage_overlay(
        base_rgb=enh_rgb,
        regions=regions,
        draw_contours=True,
        draw_boxes=True,
        draw_labels=True
    )


# ==========================================
# TOP KPI EXECUTIVE SUMMARY CARDS
# ==========================================
hhi_val = health_metrics["health_index"]
hhi_color = "#43A047" if hhi_val >= 80 else ("#FDD835" if hhi_val >= 60 else ("#FB8C00" if hhi_val >= 40 else "#E53935"))

overall_pri = health_metrics["overall_priority"]
pri_badge_class = f"badge-{overall_pri.lower()}"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {hhi_color};">{hhi_val:.1f}</div>
        <div class="metric-label">Heritage Health Index (0-100)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #7B1113;">{health_metrics['total_damaged_pct']:.1f}%</div>
        <div class="metric-label">Total Deteriorated Area</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: #2B2D42;">{len(regions)}</div>
        <div class="metric-label">Discrete Damage Zones</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div style="margin-top: 6px;"><span class="{pri_badge_class}">{overall_pri} PRIORITY</span></div>
        <div style="font-size: 0.85rem; font-weight: 600; color: #424242; margin-top: 8px;">{health_metrics['status_phrase']}</div>
        <div class="metric-label" style="margin-top: 4px;">Restoration Urgency</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==========================================
# INTERACTIVE MULTI-TAB DASHBOARD
# ==========================================
tab_visual, tab_channels, tab_inventory, tab_directives, tab_analytics = st.tabs([
    "👁️ Visual Inspection & Overlay",
    "🔬 Channel Decomposition",
    "📋 Damage Inventory & Explanations",
    "🛡️ Conservation Directives",
    "📊 Statistical Analytics"
])


# -----------------------------------------------------------------------------
# TAB 1: VISUAL INSPECTION & OVERLAY
# -----------------------------------------------------------------------------
with tab_visual:
    st.markdown("### 👁️ Photographic Evidence & Multi-Hazard Visualizations")

    view_col1, view_col2 = st.columns(2)

    with view_col1:
        st.markdown("**Original Photograph (As-Found State)**")
        st.image(orig_rgb, use_container_width=True)

        st.markdown(f"**Fused Deterioration Heatmap ({colormap_choice} Thermal)**")
        st.image(fusion_data["blended_rgb"], use_container_width=True)

    with view_col2:
        st.markdown(f"**CLAHE Contrast-Enhanced Inspection (LAB Space)**")
        st.image(enh_rgb, use_container_width=True)

        st.markdown("**Final Priority-Tagged Damage Overlay**")
        st.image(overlay_rgb, use_container_width=True)

    st.markdown("""
    **Legend of Restoration Priority Colors:**
    - 🔴 <span style="color:#E53935; font-weight:bold;">Red: CRITICAL</span> (Immediate intervention within 72 hours; deep cracks, extensive moisture or flaking)
    - 🟠 <span style="color:#FB8C00; font-weight:bold;">Orange: HIGH</span> (Urgent care within 30-60 days; advancing bio-colonies, compound damage)
    - 🟡 <span style="color:#FBC02D; font-weight:bold;">Yellow: MODERATE</span> (Scheduled remedial maintenance within 6 months; hairline fissures, superficial stains)
    - 🟢 <span style="color:#43A047; font-weight:bold;">Green: LOW</span> (Routine monitoring / stable historic patina)
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# TAB 2: CHANNEL DECOMPOSITION
# -----------------------------------------------------------------------------
with tab_channels:
    st.markdown("### 🔬 Multi-Modal IPA Channel Decomposition")
    st.caption("Detailed physical and mathematical breakdown of each individual deterioration mechanism.")

    # Channel 1: Cracks
    with st.expander("⚡ **Channel 1: Structural & Hairline Cracks**", expanded=toggle_cracks):
        c_col1, c_col2 = st.columns([1, 2])
        with c_col1:
            crack_thumb = create_channel_thumbnail(enh_rgb, crack_res["crack_mask"], (229, 57, 53))
            st.image(crack_thumb, caption="Isolated Crack Network", use_container_width=True)
        with c_col2:
            st.markdown(f"""
            - **Status:** {'Active' if toggle_cracks else 'Disabled'}
            - **Segments Count:** {crack_res['num_crack_segments']} continuous branches
            - **Total Crack Length:** ~{crack_res['total_crack_length_px']} px (Max segment: {crack_res['max_crack_length_px']} px)
            - **Estimated Average Width:** {crack_res['mean_thickness_px']} px
            - **Surface Coverage:** {crack_res['coverage_pct']}% of wall area
            - **Severity Contribution:** {crack_res['severity_score']}/100
            """)
            st.info(f"**Diagnostic Rationale:** {crack_res['explanation']}")

    # Channel 2: Moisture
    with st.expander("💧 **Channel 2: Moisture Stains & Rising Damp**", expanded=toggle_moisture):
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            moist_thumb = create_channel_thumbnail(enh_rgb, moist_res["moisture_mask"], (25, 118, 210))
            st.image(moist_thumb, caption="Isolated Moisture Plumes", use_container_width=True)
        with c_col2:
            st.markdown(f"""
            - **Status:** {'Active' if toggle_moisture else 'Disabled'}
            - **Detected Damp Plumes:** {moist_res['num_patches']} distinct zones
            - **Total Moisture Area:** {moist_res['moisture_area_pct']}% of wall area
            - **Largest Contiguous Plume:** {moist_res['largest_patch_pct']}% of wall
            - **Severity Contribution:** {moist_res['severity_score']}/100
            """)
            st.info(f"**Diagnostic Rationale:** {moist_res['explanation']}")

    # Channel 3: Peeling
    with st.expander("🍂 **Channel 3: Peeling Paint & Delaminated Plaster**", expanded=toggle_peeling):
        p_col1, p_col2 = st.columns([1, 2])
        with p_col1:
            peel_thumb = create_channel_thumbnail(enh_rgb, peel_res["peeling_mask"], (245, 124, 0))
            st.image(peel_thumb, caption="Isolated Flaking Zones", use_container_width=True)
        with p_col2:
            st.markdown(f"""
            - **Status:** {'Active' if toggle_peeling else 'Disabled'}
            - **Flaking Clusters:** {peel_res['num_peeling_patches']} delamination zones
            - **Peeling Area:** {peel_res['peeling_area_pct']}% of wall area
            - **Severity Contribution:** {peel_res['severity_score']}/100
            """)
            st.info(f"**Diagnostic Rationale:** {peel_res['explanation']}")

    # Channel 4: Biological Growth
    with st.expander("🌿 **Channel 4: Biological Growth (Moss, Algae, Lichen)**", expanded=toggle_biological):
        b_col1, b_col2 = st.columns([1, 2])
        with b_col1:
            bio_thumb = create_channel_thumbnail(enh_rgb, bio_res["biological_mask"], (56, 142, 60))
            st.image(bio_thumb, caption="Isolated Bio-Colonies", use_container_width=True)
        with b_col2:
            st.markdown(f"""
            - **Status:** {'Active' if toggle_biological else 'Disabled'}
            - **Distinct Colonies:** {bio_res['colony_count']} colonization clusters
            - **Biological Coverage:** {bio_res['coverage_pct']}% of wall area
            - **Severity Contribution:** {bio_res['severity_score']}/100
            """)
            st.info(f"**Diagnostic Rationale:** {bio_res['explanation']}")

    # Channel 5: Erosion
    with st.expander("🌪️ **Channel 5: Surface Erosion & Weathering**", expanded=toggle_erosion):
        e_col1, e_col2 = st.columns([1, 2])
        with e_col1:
            eros_thumb = create_channel_thumbnail(enh_rgb, eros_res["erosion_mask"], (121, 85, 72))
            st.image(eros_thumb, caption="Isolated Weathered Zones", use_container_width=True)
        with e_col2:
            st.markdown(f"""
            - **Status:** {'Active' if toggle_erosion else 'Disabled'}
            - **Erosion Pockets:** {eros_res['num_eroded_patches']} weathered zones
            - **Eroded Area:** {eros_res['eroded_area_pct']}% of wall area
            - **Mean Roughness Score:** {eros_res.get('mean_roughness_score', 0.0)}/100
            - **Severity Contribution:** {eros_res['severity_score']}/100
            """)
            st.info(f"**Diagnostic Rationale:** {eros_res['explanation']}")


# -----------------------------------------------------------------------------
# TAB 3: DAMAGE INVENTORY & EXPLANATIONS
# -----------------------------------------------------------------------------
with tab_inventory:
    st.markdown("### 📋 Quantitative Damage Region Inventory")
    st.caption("Each localized zone is attributed with physical dimensions, damage classification, and diagnostic reasoning.")

    if not regions:
        st.success("🎉 No significant deterioration zones detected that exceed the minimum area threshold.")
    else:
        # Build Table
        table_rows = []
        for reg in regions:
            table_rows.append({
                "Region ID": reg["region_id"],
                "Damage Classification": reg["primary_damage_name"],
                "Priority": reg["priority"],
                "Severity": f"{reg['severity_score']:.0f}/100",
                "Area (%)": f"{reg['area_pct']:.2f}%",
                "Bounding Box [X, Y, W, H]": f"[{reg['bbox'][0]}, {reg['bbox'][1]}, {reg['bbox'][2]}, {reg['bbox'][3]}]",
                "Explainable Diagnostic Rationale": reg["diagnostic_explanation"]
            })

        df_regions = pd.DataFrame(table_rows)
        st.dataframe(
            df_regions,
            column_config={
                "Region ID": st.column_config.TextColumn("Region ID", width="small"),
                "Priority": st.column_config.TextColumn("Priority", width="small"),
                "Severity": st.column_config.TextColumn("Severity", width="small"),
                "Area (%)": st.column_config.TextColumn("Area (%)", width="small"),
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### 🔍 Deep Inspection by Region")
        selected_reg_id = st.selectbox("Inspect specific region details:", [r["region_id"] for r in regions])
        selected_reg = next(r for r in regions if r["region_id"] == selected_reg_id)

        # Show cropped thumbnail of region
        bx, by, bw, bh = selected_reg["bbox"]
        # Add slight padding
        pad = 20
        y1, y2 = max(0, by - pad), min(h_img, by + bh + pad)
        x1, x2 = max(0, bx - pad), min(w_img, bx + bw + pad)
        crop_overlay = overlay_rgb[y1:y2, x1:x2]

        crop_col1, crop_col2 = st.columns([1, 2])
        with crop_col1:
            st.image(crop_overlay, caption=f"Magnified {selected_reg_id}", use_container_width=True)
        with crop_col2:
            st.markdown(f"**Classification:** {selected_reg['primary_damage_name']}")
            st.markdown(f"**Priority Tier:** <span class='badge-{selected_reg['priority'].lower()}'>{selected_reg['priority']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Severity Score:** `{selected_reg['severity_score']}/100` | **Area:** `{selected_reg['area_pct']}%` (~{selected_reg['area_px']} pixels)")
            
            ml_inf = selected_reg.get("ml_inference")
            if ml_inf and ml_inf.get("is_ml_inferred"):
                st.markdown(f"🤖 **Production ML Diagnostic:** `{ml_inf['display_name']}` (Confidence: **{ml_inf['confidence']*100:.1f}%**)")
                with st.expander("📊 Calibrated ML Probabilities"):
                    st.write(ml_inf["probabilities"])
            
            st.markdown(f"**Diagnostic Rationale:** {selected_reg['diagnostic_explanation']}")
            st.markdown(f"**Channel Overlap Analysis:**")
            st.write(selected_reg["channel_overlaps"])


# -----------------------------------------------------------------------------
# TAB 4: CONSERVATION DIRECTIVES
# -----------------------------------------------------------------------------
with tab_directives:
    st.markdown("### 🛡️ Tailored Heritage Conservation Action Plan")
    st.caption("Preservation protocols curated in accordance with ICOMOS Venice Charter, ICCROM Stone Conservation, and ASI guidelines.")

    crit_list = [r for r in regions if r["priority"] == "CRITICAL"]
    high_list = [r for r in regions if r["priority"] == "HIGH"]
    mod_list = [r for r in regions if r["priority"] == "MODERATE"]
    low_list = [r for r in regions if r["priority"] == "LOW"]

    # Critical Actions
    if crit_list:
        st.error("🚨 **CRITICAL INTERVENTIONS REQUIRED WITHIN 72 HOURS**")
        st.markdown("Active failure risks detected that threaten irreparable cultural heritage loss.")
        seen_recs = set()
        for r in crit_list:
            for rec in r.get("recommendations", []):
                if rec not in seen_recs:
                    st.markdown(f"- 🔴 **[{r['primary_damage'].upper()} - {r['region_id']}]:** {rec}")
                    seen_recs.add(rec)
        st.markdown("---")

    # High Actions
    if high_list:
        st.warning("⚠️ **HIGH PRIORITY CONSERVATION ACTIONS (SCHEDULE WITHIN 30-60 DAYS)**")
        seen_recs = set()
        for r in high_list:
            for rec in r.get("recommendations", []):
                if rec not in seen_recs:
                    st.markdown(f"- 🟠 **[{r['primary_damage'].upper()} - {r['region_id']}]:** {rec}")
                    seen_recs.add(rec)
        st.markdown("---")

    # Moderate Actions
    if mod_list:
        st.info("🟡 **PREVENTIVE REMEDIAL MAINTENANCE (SCHEDULE WITHIN 6 MONTHS)**")
        seen_recs = set()
        for r in mod_list:
            for rec in r.get("recommendations", []):
                if rec not in seen_recs:
                    st.markdown(f"- 🟡 **[{r['primary_damage'].upper()} - {r['region_id']}]:** {rec}")
                    seen_recs.add(rec)
        st.markdown("---")

    # General Charter Principles
    st.markdown("""
    #### 📜 Core Principles for Monument Conservators
    1. **Minimal Intervention:** Preserve authentic historic fabric; never over-restore or reconstruct conjectured elements.
    2. **Reversibility of Materials:** Ensure all applied consolidants (such as ethyl silicate, Paraloid B-72, or lime grouts) can be retreated in future.
    3. **Compatibility:** Never use modern Portland cement on ancient lime or mud mortar walls; differential thermal expansion and trapped salts induce violent spalling.
    4. **Environmental Control:** Remediation of roof water run-off and rising damp must precede any cosmetic or consolidation treatments.
    """)


# -----------------------------------------------------------------------------
# TAB 5: STATISTICAL ANALYTICS
# -----------------------------------------------------------------------------
with tab_analytics:
    st.markdown("### 📊 Statistical Analytics & Deterioration Profiling")

    fig_analytics = create_damage_analytics_plot(
        channel_contributions=fusion_data["channel_contributions"],
        priority_counts=health_metrics["priority_counts"],
        health_index=health_metrics["health_index"]
    )
    st.pyplot(fig_analytics, use_container_width=True)

    st.markdown("#### 📈 Summary Statistics Breakdown")
    stat_c1, stat_c2, stat_c3 = st.columns(3)
    with stat_c1:
        st.metric("Total Wall Resolution", f"{w_img} x {h_img} px")
    with stat_c2:
        st.metric("Max Localized Severity", f"{health_metrics['max_severity']:.1f} / 100")
    with stat_c3:
        st.metric("Mean Background Luminance", f"{prep_data['metrics']['mean_luminance']} / 255")


# ==========================================
# DOWNLOAD & REPORT EXPORT CENTER
# ==========================================
st.markdown("---")
st.markdown("### 📥 Download Conservation Reports & Assets")
st.caption("Generate official PDF Conservation Dossiers, machine-readable CSV inventories, and high-resolution annotated overlays.")

# Prepare Downloads
csv_bytes = generate_csv_report(regions, health_metrics)

pdf_bytes = generate_pdf_report(
    original_rgb=orig_rgb,
    enhanced_rgb=enh_rgb,
    heatmap_rgb=fusion_data["heatmap_rgb"],
    overlay_rgb=overlay_rgb,
    regions=regions,
    overall_metrics=health_metrics,
    channel_contributions=fusion_data["channel_contributions"],
    image_filename=image_name
)

# Convert overlay to PNG bytes
pil_overlay = Image.fromarray(overlay_rgb)
overlay_buf = io.BytesIO()
pil_overlay.save(overlay_buf, format="PNG")
overlay_png_bytes = overlay_buf.getvalue()

dl_col1, dl_col2, dl_col3 = st.columns(3)

with dl_col1:
    st.download_button(
        label="📄 Download PDF Conservation Dossier",
        data=pdf_bytes,
        file_name=f"KalaRaksha_Conservation_Report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )

with dl_col2:
    st.download_button(
        label="📊 Download Damage Inventory CSV",
        data=csv_bytes,
        file_name=f"KalaRaksha_Damage_Inventory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with dl_col3:
    st.download_button(
        label="🖼️ Download High-Res PNG Overlay",
        data=overlay_png_bytes,
        file_name=f"KalaRaksha_Annotated_Overlay_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        use_container_width=True
    )

# Footer & Legal Compliance Section
st.markdown("<br><hr>", unsafe_allow_html=True)

with st.expander("⚖️ Legal Compliance, Privacy Policy & DMCA Safe Harbor Disclosures"):
    st.markdown("""
    **1. Privacy & Zero-Tracking Guarantee:**
    - KalaRaksha does not collect, sell, or monetize personal identifiable information (PII).
    - Image processing occurs strictly in-memory; uploaded photographs are not persisted to persistent databases.
    - Zero remote font leakage (100% self-hosted system fonts) and zero third-party session replay trackers (no Hotjar, Mixpanel, or PostHog).

    **2. COPPA (Children's Online Privacy Protection Act) Compliance:**
    - This software is designed exclusively for professional heritage conservators, archaeologists, researchers, and users aged 13 and older. We do not knowingly collect information from minors.

    **3. DMCA Copyright & Safe Harbor Notice:**
    - Users uploading architectural or monument photography certify they have the requisite lawful rights or licenses to evaluate the asset.
    - If you believe imagery displayed or processed infringes your copyright, submit a designated DMCA Takedown Notice to our designated copyright agent at `dmca@kalaraksha.org` with asset identifier and proof of ownership.

    **4. Conservation Engineering Disclaimer:**
    - KalaRaksha is a non-destructive optical diagnostic tool. All restorative interventions (such as lime grouting, shoring, or biocide application) must be performed under the supervision of a certified structural conservation engineer in accordance with ICOMOS/ASI charters.
    """)

st.markdown(
    "<div style='text-align: center; color: #757575; font-size: 0.85rem;'>"
    "KalaRaksha © 2026 | Classical Image Processing Analysis (IPA) for Monument Preservation | "
    "Designed in accordance with ICOMOS & ASI Conservation Principles"
    "</div>",
    unsafe_allow_html=True
)
