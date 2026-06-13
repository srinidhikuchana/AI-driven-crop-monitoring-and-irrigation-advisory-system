"""
advisory.py  (enhanced)
-----------------------
Generates stage-aware, water-balance-informed irrigation advisory text.
Now incorporates:
  - ETc (crop water demand) from Hargreaves equation + crop coefficients
  - 8-day water deficit vs rainfall comparison
  - VCI / SMI indices in prompt context
  - LLM prompt updated to reference formal water-balance numbers
"""

import requests
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FREE_MODELS = [
    "meta-llama/llama-3.3-8b-instruct:free",
    "google/gemma-3-12b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

# ── Rule-based fallback ───────────────────────────────────────────────────────
STAGE_RULES = {
    "Germination": {
        "Low":    "Soil moisture is adequate. Maintain light surface irrigation every 3–4 days (10–15 mm).",
        "Medium": "Mild moisture deficit detected. Apply light irrigation of 15–20 mm immediately.",
        "High":   "⚠️ Critical moisture stress at germination stage. Apply 20–25 mm within 24 hours.",
    },
    "Vegetative": {
        "Low":    "Vegetation is well-watered. Irrigate every 5–7 days with 25–30 mm.",
        "Medium": "Moderate moisture stress. Irrigate with 30–35 mm within 48 hours.",
        "High":   "⚠️ High stress during vegetative growth. Apply 35–40 mm urgently.",
    },
    "Flowering": {
        "Low":    "Moisture acceptable. Maintain irrigation every 4–5 days (30–35 mm).",
        "Medium": "Moderate stress during flowering. Irrigate 35–40 mm within 36 hours.",
        "High":   "🔴 CRITICAL: High stress at flowering. Irrigate 40–50 mm immediately — irreversible yield loss risk.",
    },
    "Harvest-Ready": {
        "Low":    "Crop approaching maturity. Reduce irrigation — apply only 15–20 mm if wilting visible.",
        "Medium": "Light stress at harvest stage is acceptable. Apply 20–25 mm only if leaf roll observed.",
        "High":   "If >10 days to harvest apply 25–30 mm. If ≤10 days, withhold and advance maturity.",
    },
}


def _rule_based_advisory(crop, stage, stress, soil, forecast, ndvi, vv,
                          water_balance=None, vci=None, smi=None) -> str:
    base     = STAGE_RULES.get(stage, {}).get(stress, "Monitor field conditions daily.")
    rain     = forecast.get("rain_next_48h", 0)
    soil_note = soil.get("irrigation_note", "")
    temp     = forecast.get("avg_temp", 32)

    rain_note = ""
    if rain > 15:
        rain_note = f" ☔ {rain:.0f} mm rain expected in 48h — defer irrigation 2–3 days."
    elif rain > 5:
        rain_note = f" 🌦️ Light rain ({rain:.0f} mm) in 48h — reduce dose by 30%."

    temp_note = ""
    if temp > 36:
        temp_note = f" 🌡️ High temp ({temp:.0f}°C) — consider evening irrigation."

    wb_note = ""
    if water_balance:
        etc  = water_balance.get("etc_8day", 0)
        def_ = water_balance.get("deficit_mm", 0)
        irr  = water_balance.get("irr_required_mm", 0)
        wb_note = (
            f"\n\n💧 **Water Balance (8-day):** ETc = {etc:.0f} mm | "
            f"Deficit = {def_:.0f} mm | Recommended irrigation = **{irr:.0f} mm**"
        )

    vci_note = f" | VCI = {vci:.0f}/100" if vci is not None else ""
    smi_note = f" | SMI = {smi:.0f}/100" if smi is not None else ""

    return (
        f"**{crop} | {stage} Stage | Stress: {stress}**\n\n"
        f"{base}{rain_note}{temp_note}{wb_note}\n\n"
        f"🌍 **Soil:** {soil_note}\n\n"
        f"📡 **Indices:** NDVI = {ndvi:.3f} | SAR VV = {vv:.1f} dB{vci_note}{smi_note}"
    )


def _build_prompt(crop, stage, stress, soil, forecast, ndvi, vv,
                  water_balance=None, vci=None, smi=None) -> str:
    rain    = forecast.get("rain_next_48h", 0)
    avg_tmp = forecast.get("avg_temp", 32)
    soil_t  = soil.get("type", "Unknown")
    soil_dr = soil.get("drainage", "Moderate")
    soil_wr = soil.get("water_retention", "Moderate")

    wb_section = ""
    if water_balance:
        etc  = water_balance.get("etc_8day", 0)
        def_ = water_balance.get("deficit_mm", 0)
        irr  = water_balance.get("irr_required_mm", 0)
        kc   = water_balance.get("kc", 1.0)
        status = water_balance.get("status", "Unknown")
        wb_section = f"""
WATER BALANCE (8-DAY FORMAL CALCULATION):
- Crop coefficient Kc: {kc} (FAO-56)
- Reference ET0: {water_balance.get('et0_daily', 5.0):.1f} mm/day
- Crop ETc (8-day total): {etc:.1f} mm
- Rainfall (8-day): {water_balance.get('rainfall_8day', 0):.1f} mm
- Soil moisture credit: {water_balance.get('soil_credit_mm', 10):.0f} mm
- Water Deficit: {def_:.1f} mm ({status})
- Irrigation Required: {irr:.0f} mm (incl. 15% efficiency loss)
"""

    stress_indices = f"- VCI: {vci:.0f}/100 (>35 = healthy, <35 = stressed)\n" if vci is not None else ""
    stress_indices += f"- SMI: {smi:.0f}/100 (higher = wetter)\n" if smi is not None else ""

    return f"""You are an expert agricultural advisor for Indian farmers using satellite remote sensing.
Generate a concise, practical irrigation advisory grounded in the formal water balance data provided.

FIELD DATA (Sentinel-2 + Sentinel-1 SAR fusion):
- Crop: {crop}
- Growth Stage: {stage}
- Moisture Stress Level: {stress}
- NDVI: {ndvi:.3f} | SAR VV Backscatter: {vv:.1f} dB
{stress_indices}
ENVIRONMENTAL CONTEXT:
- Soil: {soil_t} | Drainage: {soil_dr} | Water Retention: {soil_wr}
- Rainfall (next 48h): {rain:.1f} mm | Average temperature (7-day): {avg_tmp:.1f}°C
{wb_section}
INSTRUCTIONS:
Write a clear, actionable advisory in 3–4 sentences. Include:
1. Irrigation decision (irrigate now / defer / skip) with exact mm from the water balance
2. One risk specific to this growth stage
3. One tip for the soil type
4. Reference the 8-day water deficit figure if provided

Keep language simple enough for a farmer with a smartphone. Be direct. No bullet points."""


def generate_advisory(crop, stage, stress, soil, forecast, ndvi, vv,
                       water_balance=None, vci=None, smi=None) -> str:
    """
    Generate irrigation advisory — LLM preferred, rule-based fallback.
    Now accepts water_balance, vci, smi for richer context.
    """
    api_key = st.secrets.get("OPENROUTER_KEY", "")

    if not api_key:
        return _rule_based_advisory(crop, stage, stress, soil, forecast, ndvi, vv,
                                     water_balance, vci, smi)

    prompt = _build_prompt(crop, stage, stress, soil, forecast, ndvi, vv,
                            water_balance, vci, smi)

    for model in FREE_MODELS:
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://agrisat-bah2026.streamlit.app",
                    "X-Title":       "AgriSat BAH 2026",
                },
                json={
                    "model":      model,
                    "max_tokens": 350,
                    "messages": [
                        {"role": "system", "content": "You are a concise agricultural advisor for Indian farmers."},
                        {"role": "user",   "content": prompt}
                    ]
                },
                timeout=20,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if content:
                return content
        except Exception:
            continue

    return _rule_based_advisory(crop, stage, stress, soil, forecast, ndvi, vv,
                                 water_balance, vci, smi)
