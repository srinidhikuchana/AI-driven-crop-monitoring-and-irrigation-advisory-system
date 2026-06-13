import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from datetime import datetime, timedelta
from utils.weather import get_weather_forecast
from utils.soil import get_soil_type
from utils.advisory import generate_advisory
from utils.classifier import predict_crop_status
from utils.sentinel import load_sentinel_scene
from utils.water_balance import compute_8day_water_deficit, generate_advisory_map

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSat — AI Crop Advisor",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }

    .hero-banner {
        background: linear-gradient(135deg, #1a2f1a 0%, #0d1f0d 50%, #0a1628 100%);
        border: 1px solid #2d5a27; border-radius: 12px;
        padding: 28px 32px; margin-bottom: 24px;
    }
    .hero-title { font-size: 2rem; font-weight: 700; color: #7dce82; margin: 0 0 6px 0; letter-spacing: -0.5px; }
    .hero-sub   { color: #8fa89b; font-size: 0.95rem; margin: 0; }

    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 18px 20px; text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
    .metric-value { color: #e6edf3; font-size: 1.5rem; font-weight: 700; }
    .metric-badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-top: 6px; }
    .badge-low    { background: #1a3a1a; color: #56d364; border: 1px solid #2ea043; }
    .badge-medium { background: #3a2a00; color: #e3b341; border: 1px solid #9e6a03; }
    .badge-high   { background: #3a0d0d; color: #f85149; border: 1px solid #da3633; }
    .badge-info   { background: #0d2137; color: #58a6ff; border: 1px solid #1f6feb; }

    .advisory-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a1f0a 100%);
        border: 1px solid #1f6feb; border-left: 4px solid #7dce82;
        border-radius: 10px; padding: 20px 24px; margin: 16px 0;
        color: #cdd9e5; font-size: 0.95rem; line-height: 1.7;
    }
    .alert-box {
        background: #1c1008; border: 1px solid #9e6a03; border-left: 4px solid #e3b341;
        border-radius: 10px; padding: 16px 20px; margin: 10px 0;
        color: #cdd9e5; font-size: 0.9rem;
    }
    .wb-box {
        background: #0d1f2d; border: 1px solid #1f6feb; border-left: 4px solid #58a6ff;
        border-radius: 10px; padding: 16px 20px; margin: 10px 0;
        color: #cdd9e5; font-size: 0.9rem;
    }
    .section-header {
        color: #7dce82; font-size: 1rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 24px 0 14px 0; padding-bottom: 8px; border-bottom: 1px solid #21262d;
    }
    .weather-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px; text-align: center;
        font-size: 0.82rem; color: #8b949e;
    }
    .weather-temp { font-size: 1.2rem; font-weight: 700; color: #e6edf3; }

    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    .stButton > button {
        background: #238636; color: white; border: none;
        border-radius: 8px; font-weight: 600; padding: 10px 24px;
        width: 100%; font-size: 0.95rem;
    }
    .stButton > button:hover { background: #2ea043; }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label { color: #8b949e !important; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ AgriSat Control Panel")
    st.markdown("---")
    st.markdown("**📍 Field Location**")

    region = st.selectbox("Quick Select Region", [
        "Custom",
        "Warangal, Telangana (Rice)",
        "Nashik, Maharashtra (Cotton)",
        "Ludhiana, Punjab (Wheat)",
        "Coimbatore, Tamil Nadu (Maize)",
        "Guntur, Andhra Pradesh (Chilli)",
    ])

    region_coords = {
        "Warangal, Telangana (Rice)":      (18.0, 79.58),
        "Nashik, Maharashtra (Cotton)":     (20.0, 73.78),
        "Ludhiana, Punjab (Wheat)":         (30.9, 75.85),
        "Coimbatore, Tamil Nadu (Maize)":   (11.0, 76.96),
        "Guntur, Andhra Pradesh (Chilli)":  (16.3, 80.44),
    }

    if region != "Custom":
        lat, lon = region_coords[region]
    else:
        lat = st.number_input("Latitude",  value=18.0, format="%.4f")
        lon = st.number_input("Longitude", value=79.58, format="%.4f")

    st.markdown("**🌾 Crop Settings**")
    crop_override = st.selectbox("Override Crop Type (optional)",
                                  ["Auto-detect", "Rice", "Wheat", "Cotton", "Maize", "Soybean"])
    scene_date = st.selectbox("Sentinel Scene",
                               ["Latest Available", "June 2025", "March 2025", "December 2024"])

    st.markdown("---")
    analyze = st.button("🔍 Analyze Field")

    st.markdown("---")
    st.markdown("**🔑 API Status**")
    owm_key  = st.secrets.get("OPENWEATHER_KEY", "")
    or_key   = st.secrets.get("OPENROUTER_KEY", "")
    cdse_id  = st.secrets.get("CDSE_CLIENT_ID", "")
    cdse_sec = st.secrets.get("CDSE_CLIENT_SECRET", "")
    st.markdown(f"{'🟢' if (cdse_id and cdse_sec) else '🟡'} Copernicus (CDSE) Sentinel data")
    st.markdown(f"{'🟢' if owm_key else '🟡'} OpenWeatherMap")
    st.markdown(f"{'🟢' if or_key  else '🟡'} OpenRouter LLM")
    st.markdown("🟢 SoilGrids ISRIC")
    if not (cdse_id and cdse_sec):
        st.caption("🟡 = using synthetic/fallback data. Add keys in secrets.toml for live data.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🛰️ AgriSat — AI Crop Monitoring & Irrigation Advisory</p>
  <p class="hero-sub">
    Sentinel-2 Optical + Sentinel-1 SAR fusion &nbsp;·&nbsp;
    ETc Water-Balance Modelling &nbsp;·&nbsp;
    VCI / SMI Stress Indices &nbsp;·&nbsp;
    Phenology-Aware Advisory &nbsp;·&nbsp;
    Bharatiya Antariksh Hackathon 2026
  </p>
</div>
""", unsafe_allow_html=True)


# ── Landing state ─────────────────────────────────────────────────────────────
if not analyze:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("2", "SAR + Optical", "Data Sources"),
        ("4", "Stage-Aware AI", "Growth Stages"),
        ("ETc", "Water Balance", "Deficit Model"),
        ("All-Weather", "Cloud-Penetrating SAR", "Coverage"),
    ]
    for col, (val, badge, label) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <span class="metric-badge badge-info">{badge}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Select a region from the sidebar and click **Analyze Field** to begin.")

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    steps = [
        ("🛰️", "Satellite Fetch", "Sentinel-2 optical + Sentinel-1 SAR with speckle filtering"),
        ("🧠", "AI Classification", "Crop type, stage, VCI/SMI stress indices, phenology"),
        ("💧", "Water Balance", "ETc = Kc × ET0 (Hargreaves). 8-day deficit vs rainfall"),
        ("🗺️", "Advisory Map", "Grid-level irrigation map for canal command areas"),
        ("🤖", "AI Advisory", "LLM-powered stage-aware recommendation with mm targets"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4, col5], steps):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:left; padding:20px;">
                <div style="font-size:1.8rem; margin-bottom:8px;">{icon}</div>
                <div style="color:#e6edf3; font-weight:600; margin-bottom:6px;">{title}</div>
                <div style="color:#8b949e; font-size:0.82rem; line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)

else:
    # ── Run analysis ──────────────────────────────────────────────────────────
    with st.spinner("🛰️ Loading Sentinel scene and running AI analysis..."):
        scene      = load_sentinel_scene(lat, lon)
        crop_input = None if crop_override == "Auto-detect" else crop_override
        result     = predict_crop_status(scene, crop_input)

    with st.spinner("🌦️ Fetching weather forecast and soil data..."):
        weather = get_weather_forecast(lat, lon)
        soil    = get_soil_type(lat, lon)

    with st.spinner("💧 Computing ETc water balance and deficit map..."):
        water_balance = compute_8day_water_deficit(
            crop=result["crop"],
            stage=result["stage"],
            lat=lat,
            weather=weather,
            soil_water_retention=soil.get("water_retention", "Moderate"),
        )
        advisory_grid, deficit_grid, _ = generate_advisory_map(
            lat=lat, lon=lon,
            ndvi_map=scene["ndvi_map"],
            crop=result["crop"],
            stage=result["stage"],
            weather=weather,
            soil_water_retention=soil.get("water_retention", "Moderate"),
        )

    with st.spinner("🤖 Generating AI irrigation advisory..."):
        advisory_text = generate_advisory(
            crop=result["crop"],
            stage=result["stage"],
            stress=result["stress_level"],
            soil=soil,
            forecast=weather,
            ndvi=result["ndvi"],
            vv=result["vv_backscatter"],
            water_balance=water_balance,
            vci=result.get("vci"),
            smi=result.get("smi"),
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Field Status Overview</div>', unsafe_allow_html=True)
    st.caption(f"📡 Data source: {scene.get('source', 'Unknown')}")

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
    stress_badge = {"Low": "badge-low", "Medium": "badge-medium", "High": "badge-high"}.get(
        result["stress_level"], "badge-info")
    wb_badge_color = {"Sufficient": "badge-low", "Mild Deficit": "badge-medium",
                      "Moderate Deficit": "badge-high", "Severe Deficit": "badge-high"}.get(
        water_balance["status"], "badge-info")

    metric_cards = [
        (k1, "Crop Type",        result["crop"],               f"{result['crop_confidence']}% conf.", "badge-info"),
        (k2, "Growth Stage",     result["stage"],              f"Stage {result['stage_num']}/4",       "badge-info"),
        (k3, "Stress Level",     result["stress_level"],       f"{result['stress_index']:.2f} index",  stress_badge),
        (k4, "VCI",              f"{result.get('vci',0):.0f}", "Veg Condition",                        "badge-info"),
        (k5, "SMI",              f"{result.get('smi',0):.0f}", "Soil Moisture",                        "badge-info"),
        (k6, "ETc 8-day",        f"{water_balance['etc_8day']:.0f} mm", f"Kc={water_balance['kc']}", "badge-info"),
        (k7, "Water Deficit",    f"{water_balance['deficit_mm']:.0f} mm", water_balance["status"],    wb_badge_color),
        (k8, "Irrig. Required",  f"{water_balance['irr_required_mm']:.0f} mm", "Apply within 8 days", "badge-medium" if water_balance['irr_required_mm'] > 0 else "badge-low"),
    ]
    for col, label, val, badge, badge_cls in metric_cards:
        with col:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="font-size:1.1rem;">{val}</div>
                <span class="metric-badge {badge_cls}">{badge}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Main layout: left maps, right advisory ────────────────────────────────
    left, right = st.columns([1.3, 1])

    with left:
        # Growth stage bar
        st.markdown('<div class="section-header">🌱 Phenology — Crop Growth Stage</div>', unsafe_allow_html=True)
        stages    = ["Germination", "Vegetative", "Flowering", "Harvest-Ready"]
        stage_num = result["stage_num"]
        ph        = result.get("phenology", {})

        fig, ax = plt.subplots(figsize=(8, 1.3))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#161b22")
        colors = ["#2ea043" if i < stage_num else "#21262d" for i in range(4)]
        for i, (s, c) in enumerate(zip(stages, colors)):
            ax.barh(0, 1, left=i, color=c, height=0.5, edgecolor="#30363d", linewidth=0.8)
            ax.text(i + 0.5, 0, s, ha="center", va="center",
                    color="#e6edf3" if i < stage_num else "#484f58",
                    fontsize=8.5, fontweight="bold" if i == stage_num - 1 else "normal")
        ax.set_xlim(0, 4)
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Phenology metrics row
        sos  = ph.get("sos_week", "N/A")
        peak = ph.get("peak_week", "N/A")
        lgp  = ph.get("lgp_weeks", "N/A")
        p_ndvi = ph.get("peak_ndvi")
        col_a, col_b, col_c = st.columns(3)
        for col, label, val in [(col_a, "SOS (weeks ago)", sos),
                                  (col_b, "Peak Growth (weeks ago)", peak),
                                  (col_c, "Length of Season (weeks)", lgp)]:
            with col:
                st.markdown(f"""<div class="metric-card" style="padding:12px;">
                    <div class="metric-label">{label}</div>
                    <div style="color:#e6edf3;font-size:1.1rem;font-weight:700;">{val}</div>
                </div>""", unsafe_allow_html=True)

        # NDVI map
        st.markdown('<div class="section-header">🗺️ NDVI Field Map</div>', unsafe_allow_html=True)
        ndvi_map = scene["ndvi_map"]
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        fig2.patch.set_facecolor("#161b22")
        ax2.set_facecolor("#161b22")
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "ndvi", ["#3a0d0d", "#9e6a03", "#2ea043", "#56d364"])
        im = ax2.imshow(ndvi_map, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        cbar = plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
        cbar.ax.yaxis.set_tick_params(color="#8b949e")
        cbar.set_label("NDVI", color="#8b949e", fontsize=9)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=8)
        ax2.set_title(f"Field NDVI — {result['crop']} ({result['stage']})",
                      color="#e6edf3", fontsize=10, pad=10)
        ax2.set_xticks([]); ax2.set_yticks([])
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # SAR map
        st.markdown('<div class="section-header">📡 SAR Backscatter Map (Sentinel-1 VV)</div>',
                    unsafe_allow_html=True)
        sar_map = scene["sar_map"]
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        fig3.patch.set_facecolor("#161b22")
        ax3.set_facecolor("#161b22")
        im3 = ax3.imshow(sar_map, cmap="Blues", aspect="auto")
        cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.03, pad=0.02)
        cbar3.ax.yaxis.set_tick_params(color="#8b949e")
        cbar3.set_label("Backscatter (dB)", color="#8b949e", fontsize=9)
        plt.setp(cbar3.ax.yaxis.get_ticklabels(), color="#8b949e", fontsize=8)
        ax3.set_title("Sentinel-1 SAR VV — Cloud-Penetrating Moisture Signal",
                      color="#e6edf3", fontsize=10, pad=10)
        ax3.set_xticks([]); ax3.set_yticks([])
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363d")
        glcm = result.get("glcm", {})
        if glcm and glcm.get("texture_contrast") is not None:
            ax3.text(0.02, 0.04,
                     f"Texture: Contrast={glcm['texture_contrast']:.0f}  "
                     f"Energy={glcm['texture_energy']:.3f}  "
                     f"Homogeneity={glcm['texture_homogeneity']:.3f}",
                     transform=ax3.transAxes, color="#8b949e", fontsize=7,
                     bbox=dict(facecolor="#161b22", alpha=0.7, edgecolor="none"))
        st.pyplot(fig3, use_container_width=True)
        plt.close()

        # ── IRRIGATION ADVISORY MAP (new) ─────────────────────────────────────
        st.markdown('<div class="section-header">🗺️ Irrigation Advisory Map — Command Area</div>',
                    unsafe_allow_html=True)
        adv_colors = ["#1a3a1a", "#9e6a03", "#c74a00", "#8b0000"]
        adv_labels = ["No irrigation", "Light (≤20 mm)", "Moderate (20–40 mm)", "Urgent (>40 mm)"]
        adv_cmap   = mcolors.ListedColormap(adv_colors)

        fig_map, ax_map = plt.subplots(figsize=(7, 4))
        fig_map.patch.set_facecolor("#161b22")
        ax_map.set_facecolor("#161b22")
        im_map = ax_map.imshow(advisory_grid, cmap=adv_cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
        ax_map.set_title(f"8-Day Irrigation Advisory — {result['crop']} (Grid Level)",
                         color="#e6edf3", fontsize=10, pad=10)
        ax_map.set_xlabel("East →", color="#8b949e", fontsize=8)
        ax_map.set_ylabel("North ↑", color="#8b949e", fontsize=8)
        ax_map.tick_params(colors="#8b949e")
        for spine in ax_map.spines.values():
            spine.set_edgecolor("#30363d")
        patches = [mpatches.Patch(facecolor=c, label=l)
                   for c, l in zip(adv_colors, adv_labels)]
        ax_map.legend(handles=patches, loc="lower right",
                      facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#cdd9e5", fontsize=7)
        st.pyplot(fig_map, use_container_width=True)
        plt.close()

    with right:
        # Advisory box
        st.markdown('<div class="section-header">💧 AI Irrigation Advisory</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="advisory-box">
            <strong style="color:#7dce82;">🤖 Stage-Aware Advisory — {result['stage']} Phase</strong><br><br>
            {advisory_text}
        </div>""", unsafe_allow_html=True)

        # ── Water Balance Panel (new) ─────────────────────────────────────────
        st.markdown('<div class="section-header">💧 8-Day Water Balance (ETc Model)</div>',
                    unsafe_allow_html=True)
        wb_status_icon = {"Sufficient": "🟢", "Mild Deficit": "🟡",
                          "Moderate Deficit": "🟠", "Severe Deficit": "🔴"}.get(
            water_balance["status"], "⚪")
        st.markdown(f"""
        <div class="wb-box">
            <strong style="color:#58a6ff;">{wb_status_icon} {water_balance['status']}</strong><br><br>
            <table style="width:100%; color:#cdd9e5; font-size:0.88rem;">
                <tr><td>Crop coefficient (Kc)</td>          <td><b>{water_balance['kc']}</b></td></tr>
                <tr><td>ET₀ (Hargreaves, daily avg)</td>    <td><b>{water_balance['et0_daily']:.1f} mm/day</b></td></tr>
                <tr><td>ETc = Kc × ET₀ (8-day total)</td>  <td><b>{water_balance['etc_8day']:.1f} mm</b></td></tr>
                <tr><td>Rainfall (8-day forecast)</td>       <td><b>{water_balance['rainfall_8day']:.1f} mm</b></td></tr>
                <tr><td>Soil moisture credit</td>            <td><b>{water_balance['soil_credit_mm']:.0f} mm</b></td></tr>
                <tr><td style="color:#f85149;">Water Deficit</td>
                    <td><b style="color:#f85149;">{water_balance['deficit_mm']:.1f} mm</b></td></tr>
                <tr><td style="color:#e3b341;">Irrigation Required</td>
                    <td><b style="color:#e3b341;">{water_balance['irr_required_mm']:.0f} mm</b></td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

        # 8-day daily breakdown
        with st.expander("📅 Daily ETc Breakdown (8-day)", expanded=False):
            wb_df = pd.DataFrame(water_balance["daily_breakdown"])
            wb_df.columns = ["Date", "ET₀ (mm)", "ETc (mm)", "Rain (mm)", "Deficit (mm)"]
            st.dataframe(wb_df, use_container_width=True, hide_index=True)

        # VCI/SMI gauges
        st.markdown('<div class="section-header">📊 Stress Indices</div>', unsafe_allow_html=True)
        vci_val = result.get("vci", 50)
        smi_val = result.get("smi", 50)

        fig_idx, axes = plt.subplots(1, 2, figsize=(6, 2))
        fig_idx.patch.set_facecolor("#161b22")
        for ax, val, label, note in [
            (axes[0], vci_val, "VCI", "Vegetation Condition Index"),
            (axes[1], smi_val, "SMI", "Soil Moisture Index"),
        ]:
            ax.set_facecolor("#161b22")
            bar_color = "#56d364" if val >= 50 else ("#e3b341" if val >= 25 else "#f85149")
            ax.barh(0, val, color=bar_color, height=0.4)
            ax.barh(0, 100 - val, left=val, color="#21262d", height=0.4)
            ax.text(50, 0.55, f"{label} = {val:.0f}/100",
                    ha="center", color="#e6edf3", fontsize=10, fontweight="bold")
            ax.text(50, -0.55, note, ha="center", color="#8b949e", fontsize=7.5)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.8, 0.8)
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_idx, use_container_width=True)
        plt.close()

        # Soil profile
        st.markdown('<div class="section-header">🌍 Soil Profile</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="color:#8b949e; font-size:0.8rem; margin-bottom:4px;">WRB Classification</div>
            <div style="color:#e6edf3; font-size:1.1rem; font-weight:600;">{soil.get('type','Unknown')}</div>
            <div style="color:#8b949e; font-size:0.8rem; margin-top:10px;">Water Retention</div>
            <div style="color:#58a6ff; font-weight:600;">{soil.get('water_retention','Moderate')}</div>
            <div style="color:#8b949e; font-size:0.8rem; margin-top:8px;">Drainage Class</div>
            <div style="color:#58a6ff; font-weight:600;">{soil.get('drainage','Moderate')}</div>
        </div>""", unsafe_allow_html=True)

        # 7-day weather
        st.markdown('<div class="section-header">🌦️ 7-Day Weather Forecast</div>',
                    unsafe_allow_html=True)
        if weather.get("daily"):
            cols = st.columns(min(7, len(weather["daily"])))
            for i, (col, day) in enumerate(zip(cols, weather["daily"][:7])):
                with col:
                    st.markdown(f"""
                    <div class="weather-card">
                        <div style="font-size:0.7rem; color:#484f58;">{day['date']}</div>
                        <div style="font-size:1.3rem;">{day['icon']}</div>
                        <div class="weather-temp">{day['temp']}°</div>
                        <div style="color:#58a6ff;">{day['rain']}mm</div>
                    </div>""", unsafe_allow_html=True)

        # Alerts
        st.markdown('<div class="section-header">⚠️ Active Alerts</div>', unsafe_allow_html=True)
        alerts = []
        if result["stress_level"] == "High":
            alerts.append(("🔴 Critical Moisture Stress",
                            f"VCI={vci_val:.0f}/100 with high SAR stress — immediate irrigation required."))
        if vci_val < 35:
            alerts.append(("🟠 VCI Below Drought Threshold",
                            f"VCI={vci_val:.0f} < 35 indicates vegetation drought stress (Kogan method)."))
        if result["stage"] == "Flowering":
            alerts.append(("🟡 Critical Growth Phase",
                            "Flowering stage: water deficit now causes permanent yield loss."))
        if water_balance["deficit_mm"] > 30:
            alerts.append(("🔴 Severe Water Deficit",
                            f"{water_balance['deficit_mm']:.0f} mm deficit over 8 days — apply {water_balance['irr_required_mm']:.0f} mm urgently."))
        if weather.get("rain_next_48h", 0) < 5:
            alerts.append(("🟡 Dry Spell Ahead",
                            "Less than 5mm rain expected in 48h. Plan irrigation accordingly."))
        if not alerts:
            alerts.append(("🟢 No Critical Alerts", "Field conditions are within normal range."))
        for title_a, body_a in alerts:
            st.markdown(f"""
            <div class="alert-box">
                <strong>{title_a}</strong><br>
                <span style="font-size:0.85rem;">{body_a}</span>
            </div>""", unsafe_allow_html=True)

    # ── NDVI trend with SOS/peak markers ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 NDVI Trend — SOS / Peak / LGP Detection</div>',
                unsafe_allow_html=True)
    dates      = pd.date_range(end=datetime.now(), periods=12, freq="2W")
    ndvi_trend = scene.get("ndvi_trend", np.random.uniform(0.2, 0.75, 12))
    ndvi_trend[-1] = result["ndvi"]
    ph         = result.get("phenology", {})

    fig4, ax4 = plt.subplots(figsize=(12, 3.5))
    fig4.patch.set_facecolor("#161b22")
    ax4.set_facecolor("#161b22")
    ax4.plot(dates, ndvi_trend, color="#7dce82", linewidth=2, marker="o",
             markersize=4, markerfacecolor="#56d364")
    ax4.fill_between(dates, ndvi_trend, alpha=0.15, color="#7dce82")
    ax4.axhline(0.3, color="#9e6a03", linestyle="--", linewidth=0.8, alpha=0.7,
                label="Stress threshold (0.30)")

    sos_idx  = ph.get("sos_week")
    peak_idx = ph.get("peak_week")
    if sos_idx is not None and isinstance(sos_idx, int):
        sos_data_idx = max(0, 11 - sos_idx // 2)
        ax4.axvline(dates[sos_data_idx], color="#58a6ff", linestyle=":", linewidth=1.2,
                    label=f"SOS (NDVI={ph.get('sos_ndvi','?')})")
    if peak_idx is not None and isinstance(peak_idx, int):
        pk_data_idx = max(0, 11 - peak_idx // 2)
        ax4.axvline(dates[pk_data_idx], color="#e3b341", linestyle=":", linewidth=1.2,
                    label=f"Peak NDVI={ph.get('peak_ndvi','?')}")

    lgp = ph.get("lgp_weeks", "?")
    ax4.set_ylabel("NDVI", color="#8b949e", fontsize=9)
    ax4.set_ylim(0, 1)
    ax4.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
    ax4.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#8b949e", fontsize=8)
    ax4.grid(axis="y", color="#21262d", linewidth=0.5)
    ax4.set_title(f"Biweekly NDVI Trend — LGP ≈ {lgp} weeks",
                  color="#e6edf3", fontsize=10, pad=8)
    plt.xticks(rotation=30, color="#8b949e")
    st.pyplot(fig4, use_container_width=True)
    plt.close()

    # ── Irrigation calendar ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">📅 8-Day Irrigation Calendar</div>',
                unsafe_allow_html=True)
    today    = datetime.now()
    cal_data = []
    for i, wb_day in enumerate(water_balance["daily_breakdown"]):
        rain      = wb_day["rain"]
        def_d     = wb_day["deficit"]
        irrigate  = def_d > 2
        amount    = round(def_d * 1.15, 0) if irrigate else 0
        cal_data.append({
            "Date":         wb_day["date"],
            "ET₀ (mm)":    wb_day["et0"],
            "ETc (mm)":    wb_day["etc"],
            "Rainfall (mm)": rain,
            "Daily Deficit": f"{def_d:.1f} mm",
            "Irrigate":      "✅ Yes" if irrigate else "⏭️ Skip",
            "Apply (mm)":    f"{amount:.0f}" if irrigate else "—",
        })
    df = pd.DataFrame(cal_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Validation / Model info ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🎯 Model Accuracy & Validation Framework</div>',
                unsafe_allow_html=True)
    acc = result.get("accuracy", {})
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("""
        <div class="metric-card" style="text-align:left; padding:18px;">
            <div style="color:#8b949e;font-size:0.8rem;margin-bottom:8px;">CLASSIFIER STATUS</div>
            <div style="color:#e3b341;font-weight:600;">Rule-Based Baseline (Hackathon)</div>
            <div style="color:#8b949e;font-size:0.82rem;margin-top:10px;">
            Production pipeline would use Random Forest / XGBoost trained on
            multi-temporal Sentinel-1/2 features. Target: <b style="color:#7dce82;">OA > 85%, κ > 0.80</b>.
            Ground-truth crop labels (field survey) required for training.
            </div>
        </div>""", unsafe_allow_html=True)
    with col_v2:
        glcm = result.get("glcm", {})
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; padding:18px;">
            <div style="color:#8b949e;font-size:0.8rem;margin-bottom:8px;">GLCM TEXTURE FEATURES (SAR)</div>
            <div style="color:#e6edf3;font-weight:600;">Sentinel-1 VV Backscatter</div>
            <div style="color:#8b949e;font-size:0.82rem;margin-top:10px;">
            Contrast: <b style="color:#58a6ff;">{glcm.get('texture_contrast','N/A')}</b>&nbsp;&nbsp;
            Energy: <b style="color:#58a6ff;">{glcm.get('texture_energy','N/A')}</b>&nbsp;&nbsp;
            Homogeneity: <b style="color:#58a6ff;">{glcm.get('texture_homogeneity','N/A')}</b><br>
            <span style="font-size:0.78rem;">Full GLCM (contrast, correlation, energy, homogeneity)
            via skimage.feature.graycomatrix in production pipeline.</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.8rem; padding:10px 0;">
        AgriSat &nbsp;·&nbsp; Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp;
        Sentinel-1 SAR + Sentinel-2 Optical Fusion &nbsp;·&nbsp;
        ETc Water Balance (Hargreaves-Samani + FAO-56 Kc) &nbsp;·&nbsp;
        VCI / SMI · SOS / Peak / LGP Phenology &nbsp;·&nbsp;
        Data: ESA Copernicus | SoilGrids ISRIC | OpenWeatherMap
    </div>""", unsafe_allow_html=True)
