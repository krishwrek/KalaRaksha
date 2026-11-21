"""
KalaRaksha - Restoration Priority & Conservation Recommendation Engine
Heritage Wall Deterioration Analysis & Restoration-Priority Mapping System

This module calculates priority levels (Critical, High, Moderate, Low) with standard
heritage conservation color codes, computes the wall-level Heritage Health Index (HHI),
and formulates actionable conservation recommendations aligned with international
heritage preservation standards (ICOMOS, ICCROM, and ASI charters).
"""

from typing import Dict, Any, List, Tuple
import numpy as np


PRIORITY_METADATA = {
    "CRITICAL": {
        "color_hex": "#E53935",
        "color_rgb": (229, 57, 53),
        "color_bgr": (53, 57, 229),
        "badge": "CRITICAL PRIORITY",
        "urgency_days": "Immediate Intervention (within 72 hours)",
        "description": "Severe structural hazard, active water ingress, or imminent loss of irreplaceable cultural fabric."
    },
    "HIGH": {
        "color_hex": "#FB8C00",
        "color_rgb": (251, 140, 0),
        "color_bgr": (0, 140, 251),
        "badge": "HIGH PRIORITY",
        "urgency_days": "Urgent Care (within 30-60 days)",
        "description": "Rapidly propagating decay, synergistic multi-hazard damage, or advancing biological colonization."
    },
    "MODERATE": {
        "color_hex": "#FDD835",
        "color_rgb": (253, 216, 53),
        "color_bgr": (53, 216, 253),
        "badge": "MODERATE PRIORITY",
        "urgency_days": "Scheduled Maintenance (within 6 months)",
        "description": "Localized superficial damage, hairline fissures, or emerging dampness requiring remedial control."
    },
    "LOW": {
        "color_hex": "#43A047",
        "color_rgb": (67, 160, 71),
        "color_bgr": (71, 160, 67),
        "badge": "LOW PRIORITY",
        "urgency_days": "Routine Monitoring (Annual Review)",
        "description": "Minor surface anomalies, stable historic patina, or early micro-weathering."
    }
}

CONSERVATION_GUIDELINES = {
    "crack": {
        "CRITICAL": [
            "Install calibrated tell-tale glass or digital caliper gauges across major fissures to record active dynamic movement.",
            "Erect temporary non-invasive structural shoring / props to arrest immediate masonry displacement.",
            "Prohibit rigid Portland cement filling; conduct non-destructive testing (ultrasonic/georadar) to map internal crack depth before hydraulic lime grouting.",
            "Consult a licensed Heritage Structural Engineer before any load-bearing intervention."
        ],
        "HIGH": [
            "Map crack propagation vectors under varying ambient temperature and humidity cycles.",
            "Clean dust and detached mortar debris using gentle dry suction micro-vacuums.",
            "Prepare compatible slaked lime (air-lime or NHL 2.0) mortar with graded river sand for micro-grouting where movement is stabilized."
        ],
        "MODERATE": [
            "Establish photographic condition baselines to monitor hairline fissure stability quarterly.",
            "Seal micro-entry points to prevent wind-driven rainwater infiltration without altering wall breathability."
        ],
        "LOW": [
            "Record fissure coordinates in the heritage GIS register; maintain periodic observational monitoring."
        ]
    },
    "moisture": {
        "CRITICAL": [
            "Immediately identify and isolate the moisture source (leaking roof eaves, drainage blockages, or ground capillary rise).",
            "Halt any application of impervious synthetic polymer coatings or waterproof membranes that trap internal damp.",
            "Allow natural passive slow evaporation; strictly avoid forced artificial hot-air dryers that cause explosive salt crystallization.",
            "Engage a conservation hydrogeologist to plan breathable French drains or sacrificial lime plinths."
        ],
        "HIGH": [
            "Apply sacrificial cellulose pulp / sepiolite clay compresses to extract sub-efflorescence soluble salts (chlorides/sulfates).",
            "Inspect perimeter drainage channels and ensure minimum 5% outward ground slope away from monument foundations.",
            "Monitor wall relative humidity and core moisture percentage using non-invasive electrical resistance probes."
        ],
        "MODERATE": [
            "Improve surrounding air circulation and ventilation to mitigate stagnant boundary-layer damp.",
            "Gently brush superficial dry salt efflorescence blooms with soft hog-bristle brushes; do not wash with tap water."
        ],
        "LOW": [
            "Conduct seasonal moisture audit during monsoon and post-monsoon transitions."
        ]
    },
    "peeling": {
        "CRITICAL": [
            "Immediately stabilize detached, curling paint/plaster flakes using micro-injection of compatible consolidant (e.g. 3-5% Paraloid B-72 in acetone or nanolime dispersion).",
            "Secure flapping flakes with temporary facing paper (Japanese tissue paper) impregnated with cyclododecane or dilute hydroxypropyl cellulose.",
            "Strictly avoid mechanical scraping, wire brushing, or adhesive tape removal."
        ],
        "HIGH": [
            "Re-adhere blistered plaster pockets using pre-wetted syringe grouting with micronized hydraulic lime slurry.",
            "Control ambient relative humidity fluctuations (maintain target 50-55% RH) to minimize hygrothermal expansion cycles.",
            "Document paint stratification and pigment binder chemistry via micro-FTIR spectroscopy."
        ],
        "MODERATE": [
            "Edge-reinforce vulnerable delamination margins with reversible lime micro-fillets to arrest flake expansion.",
            "Shelter the surface from direct wind shear and visitor proximity."
        ],
        "LOW": [
            "Maintain dusting with ultra-soft badger-hair brushes; avoid touching or physical friction."
        ]
    },
    "biological": {
        "CRITICAL": [
            "Apply targeted quaternary ammonium biocide or 2-3% benzalkonium chloride solution via cellulose poultices to devitalize deep rhizoids.",
            "Let biocide act for 72-96 hours until vegetation desiccates completely; avoid pulling live anchored moss which tears stone grains.",
            "Remediate chronic dripping water sources that nourish active biological colonies."
        ],
        "HIGH": [
            "Gently remove dead, desiccated algal biofilm using soft natural fiber brushes and deionized water misting.",
            "Clean crevice root systems using wooden spatulas (never steel scrapers).",
            "Improve sun exposure or trim excessive overarching tree canopy foliage to reduce persistent ambient humidity."
        ],
        "MODERATE": [
            "Apply preventative eco-compatible zinc-silicate or low-toxicity biocidal wash on susceptible damp shaded faces.",
            "Monitor seasonal regrowth after rainfall events."
        ],
        "LOW": [
            "Monitor stable, dormant lichen crusts which may occasionally provide bioprotective shielding if non-acidogenic."
        ]
    },
    "erosion": {
        "CRITICAL": [
            "Conduct high-density 3D photogrammetry / laser scanning to preserve sub-millimeter geometric records of crumbling relief sculptures.",
            "Apply breathable ethyl silicate / tetraethyl orthosilicate (TEOS) stone consolidant after desalinization to restore intergranular mineral cohesion.",
            "Evaluate installation of non-invasive reversible architectural canopies or protective glass micro-shelters against driving rain and sandblast."
        ],
        "HIGH": [
            "Apply sacrificial breathable lime shelter-coats to absorb exterior abrasive wind energy on vulnerable bas-reliefs.",
            "Establish visitor distance barriers (minimum 1.5m) to prevent skin lipid transfer and tactile mechanical abrasion.",
            "Test stone mineral matrix for clay swelling minerals (smectite) that aggravate spalling."
        ],
        "MODERATE": [
            "Regularly brush away windblown quartz sand deposits that act as abrasive scouring media.",
            "Monitor loss rate using micrometric erosion pins or comparative 3D point cloud subtraction."
        ],
        "LOW": [
            "Preserve natural mature patina; avoid aggressive cosmetic pressure washing."
        ]
    },
    "compound": {
        "CRITICAL": [
            "Execute a coordinated multi-phase intervention: First arrest active water ingress; second stabilize detached paint/plaster flakes; third shore dynamic structural cracks.",
            "Assemble an interdisciplinary conservation team (structural engineer, architectural conservator, materials scientist).",
            "Commission complete environmental monitoring and microclimatic data logging for the structure."
        ],
        "HIGH": [
            "Prioritize treatment sequence to avoid cross-contamination (e.g. eradicate biogrowth before applying consoliding grouts).",
            "Address moisture dissipation as prerequisite to all crack injection and plaster stabilization measures."
        ],
        "MODERATE": [
            "Treat secondary weathering factors while monitoring primary moisture/crack stability.",
            "Implement integrated preventive maintenance schedule."
        ],
        "LOW": [
            "Semi-annual holistic inspection of masonry interface."
        ]
    }
}


def classify_region_priority(region: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assign restoration priority level, color, and specific conservation actions to a single region.
    """
    sev = region.get("severity_score", 0.0)
    area_pct = region.get("area_pct", 0.0)
    primary = region.get("primary_damage", "erosion")
    is_compound = region.get("is_compound", False)
    aspect_ratio = region.get("aspect_ratio", 1.0)

    # Priority determination logic:
    # 1. Critical conditions:
    # - Very high severity score (>= 70)
    # - Large moisture area (> 3.5% of wall)
    # - Severe elongated structural cracks (aspect ratio > 6 and sev >= 60)
    # - Compound damage with sev >= 65
    # - Peeling area > 3.0% of wall with sev >= 60
    if (
        sev >= 70.0 or
        (primary == "moisture" and area_pct >= 3.0) or
        (primary == "crack" and aspect_ratio >= 6.0 and sev >= 55.0) or
        (primary == "peeling" and area_pct >= 2.5 and sev >= 55.0) or
        (is_compound and sev >= 62.0)
    ):
        priority = "CRITICAL"
    elif (
        sev >= 45.0 or
        is_compound or
        area_pct >= 1.5 or
        (primary in ("crack", "moisture") and sev >= 40.0)
    ):
        priority = "HIGH"
    elif sev >= 25.0 or area_pct >= 0.5:
        priority = "MODERATE"
    else:
        priority = "LOW"

    meta = PRIORITY_METADATA[priority]

    # Look up specific guidelines
    guideline_key = "compound" if is_compound else primary
    if guideline_key not in CONSERVATION_GUIDELINES:
        guideline_key = "erosion"

    recommendations = CONSERVATION_GUIDELINES[guideline_key].get(priority, CONSERVATION_GUIDELINES[guideline_key]["MODERATE"])

    return {
        "priority": priority,
        "color_hex": meta["color_hex"],
        "color_rgb": meta["color_rgb"],
        "color_bgr": meta["color_bgr"],
        "urgency_days": meta["urgency_days"],
        "badge": meta["badge"],
        "recommendations": recommendations
    }


def compute_heritage_health_index(
    regions: List[Dict[str, Any]],
    total_damaged_pct: float
) -> Dict[str, Any]:
    """
    Compute wall-level Overall Heritage Health Index (HHI, 0 - 100),
    overall priority rating, and executive conservation summary.

    100 = Pristine heritage fabric
    0   = Critical structural collapse / total ruin
    """
    # Count priority distributions
    counts = {"CRITICAL": 0, "HIGH": 0, "MODERATE": 0, "LOW": 0}
    max_severity = 0.0

    for reg in regions:
        pri = reg.get("priority", "LOW")
        counts[pri] = counts.get(pri, 0) + 1
        sev = reg.get("severity_score", 0.0)
        if sev > max_severity:
            max_severity = sev

    # Calculate penalty points
    # Damaged percentage penalty (max 40 pts)
    damage_penalty = min(40.0, total_damaged_pct * 2.5)

    # Priority penalties
    crit_penalty = counts["CRITICAL"] * 18.0
    high_penalty = counts["HIGH"] * 8.0
    mod_penalty = counts["MODERATE"] * 3.0
    low_penalty = counts["LOW"] * 1.0

    total_penalty = min(98.0, damage_penalty + crit_penalty + high_penalty + mod_penalty + low_penalty)
    hhi = round(max(2.0, 100.0 - total_penalty), 1)

    # Overall Wall Priority
    if counts["CRITICAL"] > 0 or hhi < 45.0 or total_damaged_pct > 15.0:
        overall_priority = "CRITICAL"
        status_phrase = "Emergency Conservation Required"
        status_color = "#E53935"
    elif counts["HIGH"] > 0 or hhi < 65.0 or total_damaged_pct > 8.0:
        overall_priority = "HIGH"
        status_phrase = "Urgent Conservation Scheduled"
        status_color = "#FB8C00"
    elif counts["MODERATE"] > 0 or hhi < 85.0 or total_damaged_pct > 3.0:
        overall_priority = "MODERATE"
        status_phrase = "Active Preventive Care Required"
        status_color = "#FDD835"
    else:
        overall_priority = "LOW"
        status_phrase = "Good Standing / Routine Maintenance"
        status_color = "#43A047"

    return {
        "health_index": hhi,
        "overall_priority": overall_priority,
        "status_phrase": status_phrase,
        "status_color": status_color,
        "priority_counts": counts,
        "total_regions": len(regions),
        "total_damaged_pct": total_damaged_pct,
        "max_severity": round(max_severity, 1)
    }
