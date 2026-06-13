"""
advisory.py
-----------
Generates stage-aware irrigation advisory text using OpenRouter LLM.

Uses free models via OpenRouter (same API pattern as GitHub Issue Solver).
Falls back to a rule-based advisory if no API key is present.

Sign up: https://openrouter.ai
Key goes in .streamlit/secrets.toml as OPENROUTER_KEY = "sk-or-..."
"""

import requests
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Best free models on OpenRouter (as of 2025) — tried in order
FREE_MODELS = [
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

# ── Rule-based fallback ───────────────────────────────────────────────────────

STAGE_RULES = {
    "Germination": {
        "Low":    "Soil moisture is adequate. Maintain light surface irrigation every 3–4 days (10–15 mm). Do not overwater — germinating seeds are sensitive to waterlogging.",
        "Medium": "Mild moisture deficit detected. Apply light irrigation of 15–20 mm immediately. Keep soil moist but not saturated during germination.",
        "High":   "⚠️ Critical moisture stress at germination stage. Apply 20–25 mm of water within 24 hours. Germination failure risk is high if deferred.",
    },
    "Vegetative": {
        "Low":    "Vegetation is well-watered. Irrigate every 5–7 days with 25–30 mm depending on evapotranspiration. Monitor NDVI for sustained greenness.",
        "Medium": "Moderate moisture stress in vegetative phase. Irrigate with 30–35 mm within 48 hours. This stage supports rapid leaf and stem development — water deficit slows canopy formation.",
        "High":   "⚠️ High stress during vegetative growth. Apply 35–40 mm urgently. Prolonged stress now reduces tiller/branch count, permanently limiting yield potential.",
    },
    "Flowering": {
        "Low":    "Moisture levels are acceptable. Flowering is the most sensitive stage — maintain consistent irrigation every 4–5 days (30–35 mm). Any sudden deficit can cause flower drop.",
        "Medium": "Moderate stress during flowering phase. Irrigate 35–40 mm within 36 hours. Water deficit during flowering causes direct grain/boll/pod set failure.",
        "High":   "🔴 CRITICAL: High moisture stress at flowering. Irrigate 40–50 mm immediately. This is the highest-impact irrigation window — delay will cause irreversible yield loss. Do not skip.",
    },
    "Harvest-Ready": {
        "Low":    "Crop approaching maturity. Reduce irrigation frequency — apply only 15–20 mm if leaves show wilting. Allow soil to dry gradually to aid harvest logistics.",
        "Medium": "Mild stress at harvest stage is acceptable. A light irrigation of 20–25 mm may be applied if the crop shows visible leaf roll. Otherwise, withhold water to advance maturity.",
        "High":   "Stress at harvest-ready stage requires assessment. If the crop is within 7–10 days of harvest, withhold irrigation. If >10 days remain, apply 25–30 mm to protect grain filling.",
    },
}


def _rule_based_advisory(crop: str, stage: str, stress: str,
                          soil: dict, forecast: dict,
                          ndvi: float, vv: float) -> str:
    """Generate a structured rule-based advisory without LLM."""
    base = STAGE_RULES.get(stage, {}).get(stress, "Monitor field conditions daily.")
    rain = forecast.get("rain_next_48h", 0)
    soil_note = soil.get("irrigation_note", "")
    temp = forecast.get("avg_temp", 32)

    rain_note = ""
    if rain > 15:
        rain_note = f" ☔ Note: {rain:.0f} mm of rainfall expected in the next 48 hours — you may defer irrigation by 2–3 days."
    elif rain > 5:
        rain_note = f" 🌦️ Light rain ({rain:.0f} mm) expected in 48h — reduce irrigation dose by 30%."

    temp_note = ""
    if temp > 36:
        temp_note = f" 🌡️ High temperature alert ({temp:.0f}°C average) — increase irrigation frequency and consider evening irrigation to reduce evapotranspiration loss."

    return (
        f"**{crop} | {stage} Stage | Stress: {stress}**\n\n"
        f"{base}{rain_note}{temp_note}\n\n"
        f"🌍 **Soil context:** {soil_note}\n\n"
        f"📡 **Satellite indices:** NDVI = {ndvi:.3f} | SAR VV = {vv:.1f} dB"
    )


# ── LLM-powered advisory ──────────────────────────────────────────────────────

def _build_prompt(crop, stage, stress, soil, forecast, ndvi, vv) -> str:
    rain    = forecast.get("rain_next_48h", 0)
    avg_tmp = forecast.get("avg_temp", 32)
    soil_t  = soil.get("type", "Unknown")
    soil_dr = soil.get("drainage", "Moderate")
    soil_wr = soil.get("water_retention", "Moderate")

    return f"""You are an expert agricultural advisor for Indian farmers.
Generate a concise, practical irrigation advisory based on satellite data and field conditions.

FIELD DATA (from Sentinel-2 optical + Sentinel-1 SAR fusion):
- Crop: {crop}
- Growth Stage: {stage}
- Moisture Stress Level: {stress}
- NDVI: {ndvi:.3f} (vegetation health index, 0–1)
- SAR VV Backscatter: {vv:.1f} dB (soil moisture proxy)

ENVIRONMENTAL CONTEXT:
- Soil Type: {soil_t} | Drainage: {soil_dr} | Water Retention: {soil_wr}
- Rainfall expected (next 48h): {rain:.1f} mm
- Average temperature (7-day): {avg_tmp:.1f}°C

INSTRUCTIONS:
Write a clear irrigation advisory in 3–4 sentences. Include:
1. Whether to irrigate now, defer, or skip
2. Exact water quantity in mm
3. One risk warning specific to this growth stage
4. One tip specific to the soil type

Keep language simple enough for a farmer with a smartphone. Do not use bullet points.
Do not repeat the input data back. Be direct and actionable."""


def generate_advisory(crop: str, stage: str, stress: str,
                      soil: dict, forecast: dict,
                      ndvi: float, vv: float) -> str:
    """
    Generate an irrigation advisory.

    Tries OpenRouter LLM first (free models), falls back to rule-based.
    """
    api_key = st.secrets.get("OPENROUTER_KEY", "")

    if not api_key:
        return _rule_based_advisory(crop, stage, stress, soil, forecast, ndvi, vv)

    prompt = _build_prompt(crop, stage, stress, soil, forecast, ndvi, vv)

    for model in FREE_MODELS:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization":  f"Bearer {api_key}",
                    "Content-Type":   "application/json",
                    "HTTP-Referer":   "https://agrisat-bah2026.streamlit.app",
                    "X-Title":        "AgriSat BAH 2026",
                },
                json={
                    "model":      model,
                    "max_tokens": 300,
                    "messages": [
                        {
                            "role":    "system",
                            "content": "You are a concise, practical agricultural advisor for Indian farmers."
                        },
                        {
                            "role":    "user",
                            "content": prompt
                        }
                    ]
                },
                timeout=20,
            )
            resp.raise_for_status()
            data    = resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            if content:
                return content

        except Exception:
            continue   # try next model

    # All models failed — use rule-based
    return _rule_based_advisory(crop, stage, stress, soil, forecast, ndvi, vv)
