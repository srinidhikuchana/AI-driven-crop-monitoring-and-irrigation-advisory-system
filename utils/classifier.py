"""
classifier.py
-------------
Multi-output crop status classifier.

Inputs : Sentinel-2 spectral indices (NDVI, NDWI, EVI, Moisture Index)
         Sentinel-1 SAR backscatter (VV, VH)

Outputs: crop_type, growth_stage, moisture_stress_level, stress_index

Architecture (production):
  - Dual-branch feature extraction
  - Branch A: [NDVI, EVI, NDWI, Moisture Index] -> optical features
  - Branch B: [VV, VH, VV-VH]                   -> SAR moisture features
  - Fusion layer -> Random Forest / XGBoost classifier
  - Three separate heads: crop type, stage, stress

For hackathon: rule-based heuristic classifier that mimics the
model's decision logic using physically-meaningful thresholds.
Replace predict_crop_status() with a trained sklearn model via
joblib.load('models/crop_stage_model.pkl') when ready.
"""

from __future__ import annotations
import numpy as np

# ── Crop type thresholds ──────────────────────────────────────────────────────
#
# Each crop has a characteristic NDVI range and SAR backscatter profile.
# Sources: FAO crop calendars + published Sentinel-1/2 fusion studies.
#
CROP_PROFILES = {
    "Rice": {
        "ndvi_range": (0.35, 0.80),
        "vv_range":   (-18, -8),
        "ndwi_min":   0.05,          # flooded fields -> positive NDWI
        "weight":     1.2,
    },
    "Wheat": {
        "ndvi_range": (0.30, 0.75),
        "vv_range":   (-16, -6),
        "ndwi_min":   -0.10,
        "weight":     1.0,
    },
    "Cotton": {
        "ndvi_range": (0.25, 0.65),
        "vv_range":   (-15, -7),
        "ndwi_min":   -0.15,
        "weight":     1.0,
    },
    "Maize": {
        "ndvi_range": (0.40, 0.85),
        "vv_range":   (-14, -5),
        "ndwi_min":   -0.05,
        "weight":     1.0,
    },
    "Soybean": {
        "ndvi_range": (0.30, 0.70),
        "vv_range":   (-17, -9),
        "ndwi_min":   -0.08,
        "weight":     0.9,
    },
}

STAGE_THRESHOLDS = {
    # (ndvi_min, ndvi_max) -> stage name, stage_num
    (0.00, 0.20): ("Germination",    1),
    (0.20, 0.45): ("Vegetative",     2),
    (0.45, 0.70): ("Flowering",      3),
    (0.70, 1.00): ("Harvest-Ready",  4),
}


def _classify_crop(ndvi: float, ndwi: float, vv: float,
                   override: str | None = None) -> tuple[str, int]:
    """Return (crop_name, confidence_pct)."""
    if override:
        prof   = CROP_PROFILES.get(override, {})
        lo, hi = prof.get("ndvi_range", (0.2, 0.8))
        conf   = int(np.clip(100 - abs(ndvi - (lo + hi) / 2) * 200, 60, 97))
        return override, conf

    scores = {}
    for crop, prof in CROP_PROFILES.items():
        lo, hi  = prof["ndvi_range"]
        vv_lo, vv_hi = prof["vv_range"]
        ndwi_ok = ndwi >= prof["ndwi_min"]

        ndvi_score = max(0, 1 - abs(ndvi - (lo + hi) / 2) / ((hi - lo) / 2))
        vv_score   = max(0, 1 - abs(vv   - (vv_lo + vv_hi) / 2) / ((vv_hi - vv_lo) / 2))
        ndwi_bonus = 0.1 if ndwi_ok else 0.0

        scores[crop] = (ndvi_score * 0.5 + vv_score * 0.4 + ndwi_bonus) * prof["weight"]

    best_crop = max(scores, key=scores.__getitem__)
    raw_conf  = scores[best_crop]
    conf = int(np.clip(raw_conf * 100, 70, 96))
    return best_crop, conf


def _classify_stage(ndvi: float) -> tuple[str, int]:
    """Return (stage_name, stage_num)."""
    for (lo, hi), (name, num) in STAGE_THRESHOLDS.items():
        if lo <= ndvi < hi:
            return name, num
    return "Harvest-Ready", 4


def _compute_stress(ndwi: float, vv: float, moisture: float | None = None) -> tuple[str, float]:
    """
    Moisture stress index (0-1), combining:
      - SAR VV backscatter (more negative -> wetter soil -> lower stress)
      - NDWI (more positive -> more surface water -> lower stress)
      - Sentinel-2 Moisture Index (B8A-B11)/(B8A+B11), higher -> more
        canopy/leaf water content -> lower stress. This is a stronger
        moisture proxy than NDWI alone and is included when available.
    """
    # Normalise VV from [-20, -5] to [0, 1]  (0 = very wet, 1 = very dry)
    vv_norm   = np.clip((vv + 20) / 15, 0, 1)
    # Normalise NDWI from [-0.3, 0.3] to [1, 0] (inverted: dry -> high)
    ndwi_norm = np.clip(1 - (ndwi + 0.3) / 0.6, 0, 1)

    if moisture is not None:
        # Moisture Index typically ranges roughly [-0.5, 0.5] over vegetation;
        # higher -> wetter canopy -> lower stress.
        moist_norm = np.clip(1 - (moisture + 0.5) / 1.0, 0, 1)
        stress_idx = float(0.40 * vv_norm + 0.25 * ndwi_norm + 0.35 * moist_norm)
    else:
        stress_idx = float(0.55 * vv_norm + 0.45 * ndwi_norm)

    if stress_idx < 0.35:
        level = "Low"
    elif stress_idx < 0.62:
        level = "Medium"
    else:
        level = "High"

    return level, round(stress_idx, 3)


# ── Public API ────────────────────────────────────────────────────────────────

def predict_crop_status(scene: dict, crop_override: str | None = None) -> dict:
    """
    Run the full classification pipeline on a loaded Sentinel scene.

    Parameters
    ----------
    scene        : dict returned by sentinel.load_sentinel_scene()
    crop_override: if set, skips crop-type detection and uses this value

    Returns
    -------
    dict with keys: crop, crop_confidence, stage, stage_num,
                    stress_level, stress_index, ndvi, vv_backscatter,
                    ndwi, moisture_index
    """
    ndvi     = scene["ndvi_mean"]
    ndwi     = scene["ndwi_mean"]
    vv       = scene["vv_mean"]
    moisture = scene.get("moisture_mean")

    crop, confidence  = _classify_crop(ndvi, ndwi, vv, crop_override)
    stage, stage_num  = _classify_stage(ndvi)
    stress_level, stress_idx = _compute_stress(ndwi, vv, moisture)

    return {
        "crop":            crop,
        "crop_confidence": confidence,
        "stage":           stage,
        "stage_num":       stage_num,
        "stress_level":    stress_level,
        "stress_index":    stress_idx,
        "ndvi":            round(ndvi, 4),
        "vv_backscatter":  round(vv, 2),
        "ndwi":            round(ndwi, 4),
        "moisture_index":  round(moisture, 4) if moisture is not None else None,
    }
