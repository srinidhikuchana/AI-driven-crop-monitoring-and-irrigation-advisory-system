"""
classifier.py  (enhanced)
--------------------------
Multi-output crop status classifier with:
  - VCI (Vegetation Condition Index) — anomaly-based stress index
  - SMI (Soil Moisture Index)        — formal soil moisture proxy
  - SOS / Peak / LGP phenology metrics derived from NDVI time-series
  - GLCM texture feature placeholders (SAR)
  - Baseline OA / Kappa scoring scaffold for future ML model

Architecture (production):
  Random Forest / XGBoost using multi-temporal Sentinel-1/2 features.
  For hackathon: rule-based heuristic that mimics model decision logic.
  Replace predict_crop_status() body with joblib.load('models/crop_model.pkl')
  when trained data is available. Accuracy target: OA > 85%.
"""

from __future__ import annotations
import numpy as np
# ── Crop profiles ─────────────────────────────────────────────────────────────
CROP_PROFILES = {
    "Rice":    {"ndvi_range": (0.35, 0.80), "vv_range": (-18, -8),  "ndwi_min":  0.05, "weight": 1.2},
    "Wheat":   {"ndvi_range": (0.30, 0.75), "vv_range": (-16, -6),  "ndwi_min": -0.10, "weight": 1.0},
    "Cotton":  {"ndvi_range": (0.25, 0.65), "vv_range": (-15, -7),  "ndwi_min": -0.15, "weight": 1.0},
    "Maize":   {"ndvi_range": (0.40, 0.85), "vv_range": (-14, -5),  "ndwi_min": -0.05, "weight": 1.0},
    "Soybean": {"ndvi_range": (0.30, 0.70), "vv_range": (-17, -9),  "ndwi_min": -0.08, "weight": 0.9},
}

STAGE_THRESHOLDS = {
    (0.00, 0.20): ("Germination",   1),
    (0.20, 0.45): ("Vegetative",    2),
    (0.45, 0.70): ("Flowering",     3),
    (0.70, 1.00): ("Harvest-Ready", 4),
}

# Historical NDVI min/max per crop (FAO / published Sentinel-2 studies)
# Used to compute VCI = (NDVI - NDVImin) / (NDVImax - NDVImin)
HISTORICAL_NDVI = {
    "Rice":    {"min": 0.10, "max": 0.85},
    "Wheat":   {"min": 0.08, "max": 0.78},
    "Cotton":  {"min": 0.08, "max": 0.70},
    "Maize":   {"min": 0.12, "max": 0.88},
    "Soybean": {"min": 0.10, "max": 0.72},
    "Unknown": {"min": 0.05, "max": 0.85},
}

# Crop coefficients (Kc) by stage — FAO-56 standard values
# Used in ETc = Kc × ET0 water-balance model
KC_TABLE = {
    "Rice":    {"Germination": 1.05, "Vegetative": 1.10, "Flowering": 1.20, "Harvest-Ready": 0.75},
    "Wheat":   {"Germination": 0.30, "Vegetative": 1.00, "Flowering": 1.10, "Harvest-Ready": 0.25},
    "Cotton":  {"Germination": 0.35, "Vegetative": 1.00, "Flowering": 1.15, "Harvest-Ready": 0.50},
    "Maize":   {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.20, "Harvest-Ready": 0.35},
    "Soybean": {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.15, "Harvest-Ready": 0.50},
    "Unknown": {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.10, "Harvest-Ready": 0.50},
}


# ─────────────────────────────────────────────────────────────────────────────
# VCI  — Vegetation Condition Index
# ─────────────────────────────────────────────────────────────────────────────
def compute_vci(ndvi: float, crop: str) -> float:
    """
    VCI = (NDVI - NDVImin) / (NDVImax - NDVImin) × 100
    Range 0–100.  Values < 35 indicate drought/stress.
    Reference: Kogan (1990) NOAA method.
    """
    hist = HISTORICAL_NDVI.get(crop, HISTORICAL_NDVI["Unknown"])
    denom = hist["max"] - hist["min"]
    if denom < 1e-6:
        return 50.0
    vci = (ndvi - hist["min"]) / denom * 100.0
    return float(np.clip(vci, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# SMI  — Soil Moisture Index
# ─────────────────────────────────────────────────────────────────────────────
def compute_smi(vv_db: float, moisture_index: float | None = None) -> float:
    """
    SMI = 0.6 × VV_norm + 0.4 × moisture_norm
    Range 0–100.  Higher = wetter soil.
    VV in dB: typical range [-20, -5] dB over agricultural land.
    """
    vv_norm = float(np.clip((vv_db + 20) / 15, 0, 1))   # -20→0, -5→1
    if moisture_index is not None:
        m_norm = float(np.clip((moisture_index + 0.5) / 1.0, 0, 1))
        smi = (0.6 * vv_norm + 0.4 * m_norm) * 100.0
    else:
        smi = vv_norm * 100.0
    return float(np.clip(smi, 0, 100))


# ─────────────────────────────────────────────────────────────────────────────
# Phenology metrics from NDVI time-series
# ─────────────────────────────────────────────────────────────────────────────
def compute_phenology_metrics(ndvi_trend: np.ndarray) -> dict:
    """
    Derive SOS, Peak, LGP from a 12-point biweekly NDVI time-series.

    SOS  (Start of Season)   : first point NDVI rises above 20% of max range
    Peak (Max Growth)        : index of maximum NDVI
    LGP  (Length of Growing Period): number of intervals above SOS threshold
    """
    if ndvi_trend is None or len(ndvi_trend) < 3:
        return {"sos_week": "N/A", "peak_week": "N/A", "lgp_weeks": "N/A",
                "sos_ndvi": None, "peak_ndvi": None}

    arr = np.array(ndvi_trend, dtype=float)
    ndvi_min = arr.min()
    ndvi_max = arr.max()
    sos_threshold = ndvi_min + 0.2 * (ndvi_max - ndvi_min)

    # SOS: first crossing upward of threshold
    sos_idx = None
    for i in range(len(arr) - 1):
        if arr[i] < sos_threshold <= arr[i + 1]:
            sos_idx = i + 1
            break
    if sos_idx is None:
        sos_idx = int(np.argmin(arr))

    # Peak: maximum value
    peak_idx = int(np.argmax(arr))

    # LGP: number of intervals >= sos_threshold
    lgp = int(np.sum(arr >= sos_threshold))

    return {
        "sos_week":   int(sos_idx * 2),      # biweekly → weeks ago
        "peak_week":  int(peak_idx * 2),
        "lgp_weeks":  int(lgp * 2),
        "sos_ndvi":   round(float(arr[sos_idx]), 3),
        "peak_ndvi":  round(float(arr[peak_idx]), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GLCM texture stub
# ─────────────────────────────────────────────────────────────────────────────
def compute_glcm_features(sar_map: np.ndarray | None) -> dict:
    """
    Compute simple GLCM-inspired texture features from SAR backscatter map.
    Full GLCM (contrast, correlation, energy, homogeneity) would use
    skimage.feature.graycomatrix in production; here we compute fast proxies.
    """
    if sar_map is None or sar_map.size == 0:
        return {"texture_contrast": None, "texture_energy": None, "texture_homogeneity": None}

    arr = sar_map.astype(np.float32)
    # Normalise to 0–255 for texture analysis
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return {"texture_contrast": 0.0, "texture_energy": 1.0, "texture_homogeneity": 1.0}
    norm = ((arr - lo) / (hi - lo) * 255).astype(np.uint8)

    # Proxy metrics
    contrast    = float(np.var(norm.astype(float)))
    energy      = float(np.sum((norm.astype(float) / 255) ** 2) / norm.size)
    homogeneity = float(1.0 / (1.0 + np.mean(np.abs(np.diff(norm.astype(float), axis=1)))))

    return {
        "texture_contrast":    round(contrast, 2),
        "texture_energy":      round(energy, 4),
        "texture_homogeneity": round(homogeneity, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Internal classifiers (unchanged from v1 but exposed for completeness)
# ─────────────────────────────────────────────────────────────────────────────
def _classify_crop(ndvi, ndwi, vv, override=None):
    if override:
        prof = CROP_PROFILES.get(override, {})
        lo, hi = prof.get("ndvi_range", (0.2, 0.8))
        conf   = int(np.clip(100 - abs(ndvi - (lo + hi) / 2) * 200, 60, 97))
        return override, conf
    scores = {}
    for crop, prof in CROP_PROFILES.items():
        lo, hi       = prof["ndvi_range"]
        vv_lo, vv_hi = prof["vv_range"]
        ndwi_ok      = ndwi >= prof["ndwi_min"]
        ndvi_score   = max(0, 1 - abs(ndvi - (lo + hi) / 2) / ((hi - lo) / 2))
        vv_score     = max(0, 1 - abs(vv - (vv_lo + vv_hi) / 2) / ((vv_hi - vv_lo) / 2))
        scores[crop] = (ndvi_score * 0.5 + vv_score * 0.4 + (0.1 if ndwi_ok else 0)) * prof["weight"]
    best = max(scores, key=scores.__getitem__)
    return best, int(np.clip(scores[best] * 100, 70, 96))


def _classify_stage(ndvi):
    for (lo, hi), (name, num) in STAGE_THRESHOLDS.items():
        if lo <= ndvi < hi:
            return name, num
    return "Harvest-Ready", 4


def _compute_stress(ndwi, vv, moisture=None):
    vv_norm   = np.clip((vv + 20) / 15, 0, 1)
    ndwi_norm = np.clip(1 - (ndwi + 0.3) / 0.6, 0, 1)
    if moisture is not None:
        moist_norm  = np.clip(1 - (moisture + 0.5) / 1.0, 0, 1)
        stress_idx  = float(0.40 * vv_norm + 0.25 * ndwi_norm + 0.35 * moist_norm)
    else:
        stress_idx  = float(0.55 * vv_norm + 0.45 * ndwi_norm)
    level = "Low" if stress_idx < 0.35 else ("Medium" if stress_idx < 0.62 else "High")
    return level, round(stress_idx, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Validation scaffold (OA / Kappa)
# ─────────────────────────────────────────────────────────────────────────────
def compute_accuracy_metrics(y_true: list, y_pred: list) -> dict:
    """
    Compute Overall Accuracy and Cohen's Kappa coefficient.
    Placeholder: when ground-truth labels are available, call this
    to validate crop-type predictions against field data.
    Target: OA > 85%.

    Parameters
    ----------
    y_true : list of ground-truth crop labels
    y_pred : list of predicted crop labels

    Returns
    -------
    dict with keys: overall_accuracy, kappa, n_samples
    """
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {"overall_accuracy": None, "kappa": None, "n_samples": 0,
                "note": "No ground truth data provided — baseline rule classifier; target OA > 85%"}

    n = len(y_true)
    classes = sorted(set(y_true) | set(y_pred))
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    oa      = correct / n

    # Cohen's Kappa
    p_e = 0.0
    for c in classes:
        p_true = y_true.count(c) / n
        p_pred = y_pred.count(c) / n
        p_e   += p_true * p_pred
    kappa = (oa - p_e) / (1 - p_e) if (1 - p_e) > 1e-8 else 0.0

    return {
        "overall_accuracy": round(oa * 100, 2),
        "kappa":            round(kappa, 4),
        "n_samples":        n,
        "note":             "Rule-based baseline. Replace with RF/XGBoost for OA > 85%."
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def predict_crop_status(scene: dict, crop_override: str | None = None) -> dict:
    """
    Full classification pipeline including VCI, SMI, phenology, texture.

    Returns
    -------
    dict with keys:
      crop, crop_confidence, stage, stage_num,
      stress_level, stress_index, ndvi, vv_backscatter, ndwi, moisture_index,
      vci, smi, phenology, glcm, kc, accuracy_note
    """
    ndvi     = scene["ndvi_mean"]
    ndwi     = scene["ndwi_mean"]
    vv       = scene["vv_mean"]
    moisture = scene.get("moisture_mean")

    crop, confidence         = _classify_crop(ndvi, ndwi, vv, crop_override)
    stage, stage_num         = _classify_stage(ndvi)
    stress_level, stress_idx = _compute_stress(ndwi, vv, moisture)

    vci      = compute_vci(ndvi, crop)
    smi      = compute_smi(vv, moisture)
    phenology = compute_phenology_metrics(scene.get("ndvi_trend"))
    glcm     = compute_glcm_features(scene.get("sar_map"))
    kc       = KC_TABLE.get(crop, KC_TABLE["Unknown"]).get(stage, 1.0)

    accuracy = compute_accuracy_metrics([], [])   # placeholder — no GT yet

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
        "vci":             round(vci, 1),
        "smi":             round(smi, 1),
        "phenology":       phenology,
        "glcm":            glcm,
        "kc":              kc,
        "accuracy":        accuracy,
    }
