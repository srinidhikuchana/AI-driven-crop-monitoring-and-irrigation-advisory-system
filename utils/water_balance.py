"""
water_balance.py
----------------
8-day crop water deficit and irrigation advisory map module.

Implements:
  - ET0 estimation via Hargreaves-Samani equation (temp + radiation proxy)
  - ETc = Kc × ET0 (crop coefficient from FAO-56)
  - 8-day water deficit = ETc − (Rainfall + effective soil moisture)
  - Grid-level advisory map generation for canal command areas

This is Objective #3 from the problem statement:
"estimate 8-day crop water deficit and generate irrigation advisory maps
for canal command areas."
"""

from __future__ import annotations
import numpy as np
from datetime import datetime, timedelta

# FAO-56 crop coefficients (same as KC_TABLE in classifier.py — shared source)
KC_TABLE = {
    "Rice":    {"Germination": 1.05, "Vegetative": 1.10, "Flowering": 1.20, "Harvest-Ready": 0.75},
    "Wheat":   {"Germination": 0.30, "Vegetative": 1.00, "Flowering": 1.10, "Harvest-Ready": 0.25},
    "Cotton":  {"Germination": 0.35, "Vegetative": 1.00, "Flowering": 1.15, "Harvest-Ready": 0.50},
    "Maize":   {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.20, "Harvest-Ready": 0.35},
    "Soybean": {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.15, "Harvest-Ready": 0.50},
    "Unknown": {"Germination": 0.40, "Vegetative": 1.00, "Flowering": 1.10, "Harvest-Ready": 0.50},
}


def hargreaves_et0(t_max: float, t_min: float, lat_deg: float,
                   doy: int | None = None) -> float:
    """
    Reference evapotranspiration (mm/day) via Hargreaves-Samani (1985).

    ET0 = 0.0023 × Ra × (T_mean + 17.8) × (T_max - T_min)^0.5

    Parameters
    ----------
    t_max    : daily max temperature (°C)
    t_min    : daily min temperature (°C)
    lat_deg  : latitude in decimal degrees
    doy      : day of year (defaults to today)

    Returns
    -------
    ET0 in mm/day
    """
    if doy is None:
        doy = datetime.now().timetuple().tm_yday

    t_mean = (t_max + t_min) / 2.0
    lat_r  = np.radians(lat_deg)

    # Solar declination and sunset hour angle
    dr    = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws    = np.arccos(-np.tan(lat_r) * np.tan(delta))

    # Extraterrestrial radiation (MJ m-2 day-1)
    Gsc  = 0.0820   # solar constant
    Ra   = (24 * 60 / np.pi) * Gsc * dr * (
        ws * np.sin(lat_r) * np.sin(delta) +
        np.cos(lat_r) * np.cos(delta) * np.sin(ws)
    )
    Ra_mm = Ra * 0.408   # convert MJ m-2 day-1 → mm/day equivalent

    et0 = 0.0023 * Ra_mm * (t_mean + 17.8) * max(t_max - t_min, 0) ** 0.5
    return max(float(et0), 0.5)   # floor at 0.5 mm/day


def compute_8day_water_deficit(
    crop: str,
    stage: str,
    lat: float,
    weather: dict,
    soil_water_retention: str = "Moderate",
) -> dict:
    """
    Compute 8-day crop water demand (ETc) and deficit.

    Parameters
    ----------
    crop                 : detected crop type
    stage                : current growth stage
    lat                  : field latitude (for ET0)
    weather              : dict from get_weather_forecast()
    soil_water_retention : from SoilGrids ('Low', 'Moderate', 'High', etc.)

    Returns
    -------
    dict with keys:
      et0_daily, etc_daily, etc_8day, rainfall_8day,
      soil_credit_mm, deficit_mm, surplus_mm, irr_required_mm,
      status, daily_breakdown
    """
    kc   = KC_TABLE.get(crop, KC_TABLE["Unknown"]).get(stage, 1.0)
    daily = (weather.get("daily") or [])[:8]

    # Soil available water credit (effective storage carrying over)
    soil_credit = {"Low": 5, "Moderate": 12, "Moderate–High": 18,
                   "High": 22, "Very High": 28}.get(
        soil_water_retention.split("/")[0].strip(), 10
    )

    daily_breakdown = []
    total_etc    = 0.0
    total_rain   = 0.0
    doy_base     = datetime.now().timetuple().tm_yday

    for i, day in enumerate(daily):
        t_max = day.get("temp", 35) + 4
        t_min = day.get("temp", 25) - 4
        rain  = day.get("rain", 0)
        et0   = hargreaves_et0(t_max, t_min, lat, doy=doy_base + i)
        etc   = et0 * kc
        total_etc  += etc
        total_rain += rain
        daily_breakdown.append({
            "date":    day.get("date", f"Day {i+1}"),
            "et0":     round(et0, 1),
            "etc":     round(etc, 1),
            "rain":    round(rain, 1),
            "deficit": round(max(etc - rain, 0), 1),
        })

    deficit_mm  = max(total_etc - total_rain - soil_credit, 0)
    surplus_mm  = max(total_rain + soil_credit - total_etc, 0)
    irr_req     = round(deficit_mm * 1.15, 1)  # 15% field-application efficiency loss

    if deficit_mm < 5:
        status = "Sufficient"
        status_color = "green"
    elif deficit_mm < 20:
        status = "Mild Deficit"
        status_color = "orange"
    elif deficit_mm < 40:
        status = "Moderate Deficit"
        status_color = "darkorange"
    else:
        status = "Severe Deficit"
        status_color = "red"

    return {
        "kc":              round(kc, 2),
        "et0_daily":       round(total_etc / max(len(daily), 1) / kc, 1),
        "etc_daily":       round(total_etc / max(len(daily), 1), 1),
        "etc_8day":        round(total_etc, 1),
        "rainfall_8day":   round(total_rain, 1),
        "soil_credit_mm":  round(soil_credit, 1),
        "deficit_mm":      round(deficit_mm, 1),
        "surplus_mm":      round(surplus_mm, 1),
        "irr_required_mm": irr_req,
        "status":          status,
        "status_color":    status_color,
        "daily_breakdown": daily_breakdown,
    }


def generate_advisory_map(
    lat: float,
    lon: float,
    ndvi_map: np.ndarray,
    crop: str,
    stage: str,
    weather: dict,
    soil_water_retention: str = "Moderate",
    grid_size: int = 8,
) -> np.ndarray:
    """
    Generate a pixel/grid-level irrigation advisory map.

    Each grid cell gets an advisory intensity score 0–3:
      0 = No irrigation needed
      1 = Mild — light irrigation
      2 = Moderate irrigation required
      3 = Urgent — high deficit

    Uses the NDVI spatial variation to modulate the field-level deficit
    across the grid, mimicking a command-area spatial output.

    Parameters
    ----------
    ndvi_map  : 2-D NDVI array (e.g. 64×64)
    grid_size : output grid resolution (default 8×8 cells)

    Returns
    -------
    advisory_grid : np.ndarray of shape (grid_size, grid_size), dtype int
    deficit_grid  : np.ndarray of shape (grid_size, grid_size), dtype float (mm)
    """
    wb    = compute_8day_water_deficit(crop, stage, lat, weather, soil_water_retention)
    base_deficit = wb["deficit_mm"]

    # Downsample NDVI map to grid
    h, w = ndvi_map.shape
    gh, gw = grid_size, grid_size
    block_h = max(h // gh, 1)
    block_w = max(w // gw, 1)

    deficit_grid  = np.zeros((gh, gw), dtype=np.float32)
    advisory_grid = np.zeros((gh, gw), dtype=np.int32)

    for i in range(gh):
        for j in range(gw):
            cell_ndvi = ndvi_map[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w].mean()
            # Lower NDVI → higher local stress → more deficit
            ndvi_factor = np.clip(1.5 - cell_ndvi, 0.5, 2.5)
            local_deficit = base_deficit * ndvi_factor
            deficit_grid[i, j] = round(float(local_deficit), 1)
            if local_deficit < 5:
                advisory_grid[i, j] = 0
            elif local_deficit < 20:
                advisory_grid[i, j] = 1
            elif local_deficit < 40:
                advisory_grid[i, j] = 2
            else:
                advisory_grid[i, j] = 3

    return advisory_grid, deficit_grid, wb
