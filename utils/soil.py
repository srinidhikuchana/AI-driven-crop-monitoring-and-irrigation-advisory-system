"""
soil.py
-------
Fetches soil classification from SoilGrids v2 (ISRIC).

No API key required — fully open REST endpoint.
Docs: https://rest.isric.org/soilgrids/v2.0/docs

Returns WRB soil classification, water retention, and drainage class
to feed into the irrigation advisory engine.
"""

import requests
import streamlit as st

# WRB → agronomic properties lookup
SOIL_PROPERTIES = {
    "Vertisols": {
        "water_retention": "Very High",
        "drainage":        "Slow",
        "irrigation_note": "Risk of waterlogging — avoid over-irrigation. Use drip/furrow."
    },
    "Alfisols": {
        "water_retention": "Moderate–High",
        "drainage":        "Moderate",
        "irrigation_note": "Good moisture retention. Monitor field capacity carefully."
    },
    "Inceptisols": {
        "water_retention": "Moderate",
        "drainage":        "Moderate",
        "irrigation_note": "Standard irrigation scheduling applicable."
    },
    "Entisols": {
        "water_retention": "Low",
        "drainage":        "Rapid",
        "irrigation_note": "Sandy soil — frequent but light irrigation needed."
    },
    "Ultisols": {
        "water_retention": "Low–Moderate",
        "drainage":        "Moderate–Rapid",
        "irrigation_note": "Leaching risk. Split irrigation doses."
    },
    "Mollisols": {
        "water_retention": "High",
        "drainage":        "Moderate",
        "irrigation_note": "Rich organic soil. Efficient water use; monitor moisture sensors."
    },
    "Oxisols": {
        "water_retention": "Low",
        "drainage":        "Rapid",
        "irrigation_note": "Highly weathered — needs frequent small irrigation events."
    },
}

# WRB group code → common name mapping
WRB_MAP = {
    "VR": "Vertisols",
    "AL": "Alfisols",
    "IC": "Inceptisols",
    "ET": "Entisols",
    "UT": "Ultisols",
    "ML": "Mollisols",
    "OX": "Oxisols",
    "CM": "Cambisols",
    "LV": "Luvisols",
    "PH": "Phaeozems",
    "FL": "Fluvisols",
    "RG": "Regosols",
    "LP": "Leptosols",
    "AR": "Arenosols",
}

# Fallback based on Indian geomorphological zones
INDIA_SOIL_DEFAULTS = {
    # (lat_min, lat_max, lon_min, lon_max): soil_type
    (8,  18, 73,  82): "Vertisols",      # Deccan Plateau (black cotton soil)
    (18, 30, 73,  85): "Inceptisols",    # Indo-Gangetic Plain
    (28, 35, 74,  80): "Mollisols",      # Punjab / Haryana
    (8,  15, 76,  80): "Alfisols",       # Eastern Ghats
    (22, 28, 68,  74): "Entisols",       # Rajasthan desert fringe
}


def _lookup_india_default(lat: float, lon: float) -> str:
    for (la, lb, lo1, lo2), soil in INDIA_SOIL_DEFAULTS.items():
        if la <= lat <= lb and lo1 <= lon <= lo2:
            return soil
    return "Inceptisols"  # Generic fallback


def get_soil_type(lat: float, lon: float) -> dict:
    """
    Query SoilGrids v2 for WRB soil classification at (lat, lon).
    Falls back to India regional defaults if the API is unavailable.
    """
    try:
        url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}"
            f"&property=wrb"
            f"&depth=0-5cm"
            f"&value=mean"
        )
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json()

        # Parse WRB from response
        wrb_code = None
        try:
            layers = data["properties"]["layers"]
            for layer in layers:
                if "wrb" in layer.get("name", "").lower():
                    # WRB returns a most probable group
                    vals = layer.get("depths", [{}])[0].get("values", {})
                    wrb_code = vals.get("mean", None)
                    break
        except (KeyError, IndexError, TypeError):
            pass

        soil_name = WRB_MAP.get(str(wrb_code), _lookup_india_default(lat, lon))
        props = SOIL_PROPERTIES.get(soil_name, {
            "water_retention": "Moderate",
            "drainage":        "Moderate",
            "irrigation_note": "Standard irrigation scheduling applicable."
        })

        return {
            "type":            soil_name,
            "wrb_code":        wrb_code,
            "water_retention": props["water_retention"],
            "drainage":        props["drainage"],
            "irrigation_note": props["irrigation_note"],
            "source":          "SoilGrids v2 ISRIC",
        }

    except Exception:
        # Silently fall back — no error shown to user
        soil_name = _lookup_india_default(lat, lon)
        props = SOIL_PROPERTIES.get(soil_name, {
            "water_retention": "Moderate",
            "drainage":        "Moderate",
            "irrigation_note": "Standard scheduling applicable."
        })
        return {
            "type":            soil_name,
            "wrb_code":        None,
            "water_retention": props["water_retention"],
            "drainage":        props["drainage"],
            "irrigation_note": props["irrigation_note"],
            "source":          "Regional default (India geomorphology)",
        }
