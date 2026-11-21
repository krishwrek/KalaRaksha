"""
KalaRaksha - Heritage Wall Deterioration Analysis & Restoration Mapping Package
Core Image Processing Analysis (IPA) Modules
"""

from .preprocessing import preprocess_heritage_image
from .crack_detection import detect_cracks
from .moisture_detection import detect_moisture_seepage
from .peeling_detection import detect_peeling_plaster
from .biological_growth import detect_biological_growth
from .erosion_detection import detect_surface_erosion
from .heatmap_fusion import fuse_deterioration_channels
from .region_analysis import analyze_damage_regions
from .priority_engine import classify_region_priority, compute_heritage_health_index
from .visualization import draw_damage_overlay, create_damage_analytics_plot, create_channel_thumbnail
from .report_generator import generate_csv_report, generate_pdf_report
from .security import (
    validate_image_payload,
    sanitize_filename,
    InMemoryRateLimiter,
    verify_api_key,
    get_production_security_headers,
    safe_html
)

__all__ = [
    "preprocess_heritage_image",
    "detect_cracks",
    "detect_moisture_seepage",
    "detect_peeling_plaster",
    "detect_biological_growth",
    "detect_surface_erosion",
    "fuse_deterioration_channels",
    "analyze_damage_regions",
    "classify_region_priority",
    "compute_heritage_health_index",
    "draw_damage_overlay",
    "create_damage_analytics_plot",
    "create_channel_thumbnail",
    "generate_csv_report",
    "generate_pdf_report",
    "validate_image_payload",
    "sanitize_filename",
    "InMemoryRateLimiter",
    "verify_api_key",
    "get_production_security_headers",
    "safe_html",
]
