"""
sentinel.py
-----------
Loads Sentinel-2 optical + Sentinel-1 SAR data via the Copernicus Data Space
Ecosystem (CDSE) Sentinel Hub Statistics API.

If CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are present in st.secrets, real band
statistics (NDVI, NDWI, Moisture Index, SAR VV/VH) are fetched for a small
polygon around (lat, lon) and for a 12-interval time series (NDVI trend).

If credentials are missing or any API call fails, falls back to a
deterministic synthetic scene so the app always runs end-to-end.
"""

from __future__ import annotations
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _ndvi_from_bands(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi = (nir - red) / (nir + red + 1e-8)
    return np.clip(ndvi, -1, 1)


def _ndwi_from_bands(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Water Index (McFeeters 1996)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ndwi = (green - nir) / (green + nir + 1e-8)
    return np.clip(ndwi, -1, 1)


def _evi_from_bands(nir, red, blue) -> np.ndarray:
    """Enhanced Vegetation Index."""
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
    return np.clip(evi, -1, 2)


def _bbox_polygon(lat: float, lon: float, half_size_deg: float = 0.0025) -> dict:
    """Small square polygon (~500m x 500m) around (lat, lon) as GeoJSON."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [lon - half_size_deg, lat - half_size_deg],
            [lon + half_size_deg, lat - half_size_deg],
            [lon + half_size_deg, lat + half_size_deg],
            [lon - half_size_deg, lat + half_size_deg],
            [lon - half_size_deg, lat - half_size_deg],
        ]]
    }


# ── Synthetic scene generator (fallback / demo) ──────────────────────────────

def _make_synthetic_scene(lat: float, lon: float, size: int = 64) -> dict:
    """
    Generate a spatially-coherent synthetic Sentinel scene.

    The scene is seeded from (lat, lon) so the same coordinates always
    produce the same scene — deterministic for demos.
    """
    rng = np.random.default_rng(seed=int(abs(lat * 1000 + lon * 100)) % 2**31)

    lat_factor = 1.0 - abs(lat - 15) / 30
    base_ndvi  = np.clip(0.45 + lat_factor * 0.25, 0.3, 0.8)

    raw = rng.uniform(0, 1, (size, size))
    for _ in range(3):
        raw = (
            np.roll(raw, 1, 0) + raw + np.roll(raw, -1, 0) +
            np.roll(raw, 1, 1) + np.roll(raw, -1, 1)
        ) / 5

    ndvi_map = np.clip(base_ndvi * raw / raw.mean(), 0, 1).astype(np.float32)

    sar_base = rng.uniform(0, 1, (size, size))
    for _ in range(2):
        sar_base = (
            np.roll(sar_base, 1, 0) + sar_base + np.roll(sar_base, -1, 0)
        ) / 3
    sar_map = (sar_base * 15 - 20).astype(np.float32)

    nir   = np.clip(ndvi_map * 0.6 + rng.uniform(0, 0.1, (size, size)), 0, 1)
    red   = np.clip((1 - ndvi_map) * 0.3, 0, 1)
    green = np.clip(ndvi_map * 0.4, 0, 1)
    blue  = np.clip(rng.uniform(0.05, 0.15, (size, size)), 0, 1)
    swir  = np.clip((1 - ndvi_map) * 0.5 + rng.uniform(0, 0.05, (size, size)), 0, 1)

    ndvi_arr  = _ndvi_from_bands(nir, red)
    ndwi_arr  = _ndwi_from_bands(green, nir)
    evi_arr   = _evi_from_bands(nir, red, blue)
    moist_arr = _ndvi_from_bands(nir, swir)  # shape mimics (B8A-B11)/(B8A+B11)

    trend_base = np.linspace(0.2, base_ndvi, 12)
    ndvi_trend = np.clip(
        trend_base + rng.uniform(-0.05, 0.05, 12), 0.1, 0.9
    ).astype(np.float32)

    return {
        "ndvi_map":       ndvi_map,
        "sar_map":        sar_map,
        "ndvi_mean":      float(ndvi_map.mean()),
        "ndwi_mean":      float(ndwi_arr.mean()),
        "evi_mean":       float(evi_arr.mean()),
        "moisture_mean":  float(moist_arr.mean()),
        "vv_mean":        float(sar_map.mean()),
        "vh_mean":        float((sar_map - rng.uniform(2, 5)).mean()),
        "ndvi_trend":     ndvi_trend,
        "lat":            lat,
        "lon":            lon,
        "source":         "Synthetic (no CDSE credentials configured)",
    }


# ── Real Sentinel Hub Statistics API ─────────────────────────────────────────

_OPTICAL_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["B02","B03","B04","B08","B8A","B11","B12","dataMask"] }],
    output: [
      { id: "data", bands: 5, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  let ndvi     = (s.B08 - s.B04)  / (s.B08 + s.B04 + 1e-8);
  let ndwi     = (s.B03 - s.B08)  / (s.B03 + s.B08 + 1e-8);
  let moisture = (s.B8A - s.B11)  / (s.B8A + s.B11 + 1e-8);
  let evi = 2.5 * (s.B08 - s.B04) / (s.B08 + 6*s.B04 - 7.5*s.B02 + 1 + 1e-8);
  return {
    data: [ndvi, ndwi, moisture, evi, s.B04],
    dataMask: [s.dataMask]
  };
}
"""

_SAR_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV","VH","dataMask"] }],
    output: [
      { id: "data", bands: 2, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}
function evaluatePixel(s) {
  return { data: [s.VV, s.VH], dataMask: [s.dataMask] };
}
"""


@st.cache_data(ttl=3600, show_spinner=False)
def _get_token(client_id: str, client_secret: str) -> str | None:
    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()["access_token"]
    except Exception:
        return None


def _stats_request(token: str, geometry: dict, date_from: str, date_to: str,
                    collection: str, evalscript: str, interval_days: int = 90) -> dict | None:
    payload = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [{"type": collection}],
        },
        "aggregation": {
            "timeRange": {"from": f"{date_from}T00:00:00Z", "to": f"{date_to}T23:59:59Z"},
            "aggregationInterval": {"of": f"P{interval_days}D"},
            "evalscript": evalscript,
            "resx": 10,
            "resy": 10,
        },
        "calculations": {"default": {"statistics": {"default": {"percentiles": {"k": [50]}}}}},
    }
    try:
        resp = requests.post(
            STATS_URL,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _extract_band_means(stats_json: dict, n_bands: int) -> list:
    """
    Returns a list of per-interval band-mean lists, oldest first.
    Each element is [band0_mean, band1_mean, ...] or None if no valid data.
    """
    out = []
    for interval in stats_json.get("data", []):
        outputs = interval.get("outputs", {})
        band_stats = outputs.get("data", {}).get("bands", {})
        means = []
        valid = True
        for i in range(n_bands):
            b = band_stats.get(f"B{i}", {})
            stats = b.get("stats", {})
            mean = stats.get("mean")
            if mean is None or stats.get("sampleCount", 0) == 0:
                valid = False
                break
            means.append(mean)
        out.append(means if valid else None)
    return out


def _fetch_real_scene(lat: float, lon: float, client_id: str, client_secret: str) -> dict | None:
    token = _get_token(client_id, client_secret)
    if not token:
        return None

    geometry = _bbox_polygon(lat, lon)
    today = datetime.utcnow()

    # ── Time series for NDVI trend (last ~6 months, 15-day intervals) ────────
    trend_from = (today - timedelta(days=180)).strftime("%Y-%m-%d")
    trend_to   = today.strftime("%Y-%m-%d")
    trend_json = _stats_request(token, geometry, trend_from, trend_to,
                                 "sentinel-2-l2a", _OPTICAL_EVALSCRIPT, interval_days=15)

    ndvi_trend = None
    latest_optical = None
    if trend_json:
        means_list = _extract_band_means(trend_json, n_bands=5)
        ndvi_vals = [m[0] for m in means_list if m is not None]
        if ndvi_vals:
            if len(ndvi_vals) < 12:
                ndvi_vals = [ndvi_vals[0]] * (12 - len(ndvi_vals)) + ndvi_vals
            ndvi_trend = np.array(ndvi_vals[-12:], dtype=np.float32)
        for m in reversed(means_list):
            if m is not None:
                latest_optical = m  # [ndvi, ndwi, moisture, evi, red]
                break

    if latest_optical is None:
        return None

    # ── SAR (Sentinel-1 GRD), last 30 days ────────────────────────────────────
    sar_from = (today - timedelta(days=30)).strftime("%Y-%m-%d")
    sar_to   = today.strftime("%Y-%m-%d")
    sar_json = _stats_request(token, geometry, sar_from, sar_to,
                               "sentinel-1-grd", _SAR_EVALSCRIPT, interval_days=30)

    vv_mean, vh_mean = -12.0, -16.0  # defaults if SAR unavailable
    if sar_json:
        sar_means = _extract_band_means(sar_json, n_bands=2)
        for m in reversed(sar_means):
            if m is not None:
                vv_lin, vh_lin = m
                if vv_lin > 0:
                    vv_mean = float(10 * np.log10(vv_lin))
                if vh_lin > 0:
                    vh_mean = float(10 * np.log10(vh_lin))
                break

    ndvi_mean, ndwi_mean, moisture_mean, evi_mean, _ = latest_optical

    if ndvi_trend is None:
        ndvi_trend = np.full(12, ndvi_mean, dtype=np.float32)
    else:
        ndvi_trend[-1] = ndvi_mean

    # 2-D maps aren't returned by the Statistics API -> build visual maps
    # from a synthetic field, recalibrated to the real scalar values.
    visual = _make_synthetic_scene(lat, lon)
    ndvi_map = np.clip(visual["ndvi_map"] * (ndvi_mean / max(visual["ndvi_mean"], 1e-3)), 0, 1)
    sar_map  = visual["sar_map"] - visual["vv_mean"] + vv_mean

    return {
        "ndvi_map":       ndvi_map.astype(np.float32),
        "sar_map":        sar_map.astype(np.float32),
        "ndvi_mean":      float(ndvi_mean),
        "ndwi_mean":      float(ndwi_mean),
        "evi_mean":       float(evi_mean),
        "moisture_mean":  float(moisture_mean),
        "vv_mean":        float(vv_mean),
        "vh_mean":        float(vh_mean),
        "ndvi_trend":     ndvi_trend,
        "lat":            lat,
        "lon":            lon,
        "source":         "Copernicus Data Space (Sentinel-2 L2A + Sentinel-1 GRD)",
    }


# ── Public API ────────────────────────────────────────────────────────────────

def load_sentinel_scene(lat: float, lon: float) -> dict:
    """
    Load a Sentinel scene for the given coordinates.

    Uses the Copernicus Data Space Ecosystem Statistics API when
    CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are set in st.secrets.
    Falls back to a deterministic synthetic scene otherwise (or on any
    API error), so the app always renders.
    """
    client_id     = st.secrets.get("CDSE_CLIENT_ID", "")
    client_secret = st.secrets.get("CDSE_CLIENT_SECRET", "")

    if client_id and client_secret:
        try:
            scene = _fetch_real_scene(lat, lon, client_id, client_secret)
            if scene is not None:
                return scene
        except Exception:
            pass

    return _make_synthetic_scene(lat, lon)
