import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import json
from datetime import datetime, timedelta
from utils.weather import get_weather_forecast
from utils.soil import get_soil_type
from utils.advisory import generate_advisory
from utils.classifier import predict_crop_status
from utils.sentinel import load_sentinel_scene

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriSat — AI Crop Advisor",
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
        border: 1px solid #2d5a27;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #7dce82;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #8fa89b;
        font-size: 0.95rem;
        margin: 0;
    }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px 20px;
        text-align: center;
    }
    .metric-label {
        color: #8b949e;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #e6edf3;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .metric-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 6px;
    }
    .badge-low    { background: #1a3a1a; color: #56d364; border: 1px solid #2ea043; }
    .badge-medium { background: #3a2a00; color: #e3b341; border: 1px solid #9e6a03; }
    .badge-high   { background: #3a0d0d; color: #f85149; border: 1px solid #da3633; }
    .badge-info   { background: #0d2137; color: #58a6ff; border: 1px solid #1f6feb; }

    .advisory-box {
        background: linear-gradient(135deg, #0d2137 0%, #0a1f0a 100%);
        border: 1px solid #1f6feb;
        border-left: 4px solid #7dce82;
        border-radius: 10px;
        padding: 20px 24px;
        margin: 16px 0;
        color: #cdd9e5;
        font-size: 0.95rem;
        line-height: 1.7;
    }
    .alert-box {
        background: #1c1008;
        border: 1px solid #9e6a03;
        border-left: 4px solid #e3b341;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 10px 0;
        color: #cdd9e5;
        font-size: 0.9rem;
    }
    .section-header {
        color: #7dce82;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 14px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
    }
    .stage-bar-wrap { margin: 10px 0 20px 0; }
    .stage-label {
        color: #8b949e;
        font-size: 0.78rem;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .weather-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        font-size: 0.82rem;
        color: #8b949e;
    }
    .weather-temp { font-size: 1.2rem; font-weight: 700; color: #e6edf3; }
    
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
    .stButton > button {
        background: #238636;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 24px;
        width: 100%;
        font-size: 0.95rem;
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
    
    # Preset Indian farming regions
    region = st.selectbox("Quick Select Region", [
        "Custom",
        "Warangal, Telangana (Rice)",
        "Nashik, Maharashtra (Cotton)",
        "Ludhiana, Punjab (Wheat)",
        "Coimbatore, Tamil Nadu (Maize)",
        "Guntur, Andhra Pradesh (Chilli)"
    ])

    region_coords = {
        "Warangal, Telangana (Rice)":        (18.0, 79.58),
        "Nashik, Maharashtra (Cotton)":       (20.0, 73.78),
        "Ludhiana, Punjab (Wheat)":           (30.9, 75.85),
        "Coimbatore, Tamil Nadu (Maize)":     (11.0, 76.96),
        "Guntur, Andhra Pradesh (Chilli)":    (16.3, 80.44),
    }

    if region != "Custom":
        lat, lon = region_coords[region]
    else:
        lat = st.number_input("Latitude",  value=18.0, format="%.4f")
        lon = st.number_input("Longitude", value=79.58, format="%.4f")

    st.markdown("**🌾 Crop Settings**")
    crop_override = st.selectbox("Override Crop Type (optional)", [
        "Auto-detect", "Rice", "Wheat", "Cotton", "Maize", "Soybean"
    ])
    scene_date = st.selectbox("Sentinel Scene", [
        "Latest Available", "June 2025", "March 2025", "December 2024"
    ])

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
    st.markdown(f"🟢 SoilGrids ISRIC")
    if not (cdse_id and cdse_sec):
        st.caption("🟡 = using synthetic / fallback data. Add keys in secrets.toml for live data.")


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🛰️ AgriSat — AI Crop Monitoring & Irrigation Advisory</p>
  <p class="hero-sub">
    Sentinel-2 Optical + Sentinel-1 SAR fusion &nbsp;·&nbsp; 
    All-weather monitoring &nbsp;·&nbsp; 
    Stage-aware irrigation advisory &nbsp;·&nbsp;
    Bharatiya Antariksh Hackathon 2026
  </p>
</div>
""", unsafe_allow_html=True)


# ── Main logic ────────────────────────────────────────────────────────────────
if not analyze:
    # Landing state
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Data Sources</div>
            <div class="metric-value">2</div>
            <span class="metric-badge badge-info">SAR + Optical</span>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Growth Stages</div>
            <div class="metric-value">4</div>
            <span class="metric-badge badge-info">Stage-Aware AI</span>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Coverage</div>
            <div class="metric-value">All-Weather</div>
            <span class="metric-badge badge-info">Cloud-Penetrating SAR</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.info("👈 Select a region from the sidebar and click **Analyze Field** to begin.")

    st.markdown('<div class="section-header">How It Works</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    steps = [
        ("🛰️", "Satellite Fetch", "Pulls Sentinel-2 optical + Sentinel-1 SAR data for your field"),
        ("🧠", "AI Classification", "Detects crop type, growth stage, and moisture stress index"),
        ("🌦️", "Weather + Soil", "Integrates 7-day forecast and SoilGrids soil classification"),
        ("💧", "Advisory", "Generates stage-aware, LLM-powered irrigation recommendations"),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3,col4], steps):
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

    with st.spinner("🤖 Generating AI irrigation advisory..."):
        advisory_text = generate_advisory(
            crop=result["crop"],
            stage=result["stage"],
            stress=result["stress_level"],
            soil=soil,
            forecast=weather,
            ndvi=result["ndvi"],
            vv=result["vv_backscatter"]
        )

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Field Status Overview</div>', unsafe_allow_html=True)
    st.caption(f"📡 Data source: {scene.get('source', 'Unknown')}")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    stress_badge = {
        "Low":    "badge-low",
        "Medium": "badge-medium",
        "High":   "badge-high"
    }.get(result["stress_level"], "badge-info")

    with k1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Crop Type</div>
            <div class="metric-value">{result['crop']}</div>
            <span class="metric-badge badge-info">{result['crop_confidence']}% conf.</span>
        </div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Growth Stage</div>
            <div class="metric-value" style="font-size:1.1rem;">{result['stage']}</div>
            <span class="metric-badge badge-info">Stage {result['stage_num']}/4</span>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Moisture Stress</div>
            <div class="metric-value">{result['stress_level']}</div>
            <span class="metric-badge {stress_badge}">{result['stress_index']:.2f} index</span>
        </div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">NDVI</div>
            <div class="metric-value">{result['ndvi']:.3f}</div>
            <span class="metric-badge badge-info">Vegetation Health</span>
        </div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">SAR Backscatter</div>
            <div class="metric-value">{result['vv_backscatter']:.1f}</div>
            <span class="metric-badge badge-info">VV dB</span>
        </div>""", unsafe_allow_html=True)
    with k6:
        moist_val = result.get("moisture_index")
        moist_display = f"{moist_val:.3f}" if moist_val is not None else "N/A"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Moisture Index</div>
            <div class="metric-value">{moist_display}</div>
            <span class="metric-badge badge-info">(B8A-B11)/(B8A+B11)</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Two-column layout ─────────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        # Growth stage timeline
        st.markdown('<div class="section-header">🌱 Crop Growth Stage</div>', unsafe_allow_html=True)
        stages     = ["Germination", "Vegetative", "Flowering", "Harvest-Ready"]
        stage_num  = result["stage_num"]

        fig, ax = plt.subplots(figsize=(7, 1.2))
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

        # NDVI heatmap
        st.markdown('<div class="section-header">🗺️ NDVI Field Map (Simulated)</div>', unsafe_allow_html=True)
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

        # SAR backscatter
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
        st.pyplot(fig3, use_container_width=True)
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

        # Soil info
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
                           f"NDVI {result['ndvi']:.2f} with high SAR stress index — immediate irrigation required."))
        if result["stage"] == "Flowering":
            alerts.append(("🟡 Critical Growth Phase",
                           "Crop is in flowering stage. Water deficit now causes permanent yield loss."))
        if weather.get("rain_next_48h", 0) < 5:
            alerts.append(("🟡 Dry Spell Ahead",
                           f"Less than 5mm rain expected in 48h. Plan irrigation accordingly."))
        if not alerts:
            alerts.append(("🟢 No Critical Alerts", "Field conditions are within normal range."))
        for title_a, body_a in alerts:
            st.markdown(f"""
            <div class="alert-box">
                <strong>{title_a}</strong><br>
                <span style="font-size:0.85rem;">{body_a}</span>
            </div>""", unsafe_allow_html=True)

        # 7-day irrigation calendar
        st.markdown('<div class="section-header">📅 Irrigation Calendar</div>',
                    unsafe_allow_html=True)
        today = datetime.now()
        cal_data = []
        for i in range(7):
            d = today + timedelta(days=i)
            rain = weather["daily"][i]["rain"] if weather.get("daily") and i < len(weather["daily"]) else 0
            irrigate = rain < 5 and (i % 2 == 0 or result["stress_level"] == "High")
            amount   = max(0, 30 - rain * 3) if irrigate else 0
            cal_data.append({
                "Date":      d.strftime("%b %d"),
                "Rainfall":  f"{rain:.1f} mm",
                "Irrigate":  "✅ Yes" if irrigate else "⏭️ Skip",
                "Amount":    f"{amount:.0f} mm" if irrigate else "—"
            })
        df = pd.DataFrame(cal_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── NDVI trend chart ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 NDVI Historical Trend</div>',
                unsafe_allow_html=True)
    dates = pd.date_range(end=datetime.now(), periods=12, freq="2W")
    ndvi_trend = scene.get("ndvi_trend", np.random.uniform(0.2, 0.75, 12))
    ndvi_trend[-1] = result["ndvi"]

    fig4, ax4 = plt.subplots(figsize=(12, 3))
    fig4.patch.set_facecolor("#161b22")
    ax4.set_facecolor("#161b22")
    ax4.plot(dates, ndvi_trend, color="#7dce82", linewidth=2, marker="o",
             markersize=4, markerfacecolor="#56d364")
    ax4.fill_between(dates, ndvi_trend, alpha=0.15, color="#7dce82")
    ax4.axhline(0.3, color="#9e6a03", linestyle="--", linewidth=0.8, alpha=0.7,
                label="Stress threshold")
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("NDVI", color="#8b949e", fontsize=9)
    ax4.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
    ax4.set_facecolor("#161b22")
    ax4.legend(facecolor="#161b22", edgecolor="#30363d",
               labelcolor="#8b949e", fontsize=8)
    ax4.grid(axis="y", color="#21262d", linewidth=0.5)
    plt.xticks(rotation=30, color="#8b949e")
    st.pyplot(fig4, use_container_width=True)
    plt.close()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#484f58; font-size:0.8rem; padding:10px 0;">
        AgriSat &nbsp;·&nbsp; Bharatiya Antariksh Hackathon 2026 &nbsp;·&nbsp;
        Sentinel-1 SAR + Sentinel-2 Optical Fusion &nbsp;·&nbsp;
        Data: ESA Copernicus | SoilGrids ISRIC | OpenWeatherMap
    </div>""", unsafe_allow_html=True)
