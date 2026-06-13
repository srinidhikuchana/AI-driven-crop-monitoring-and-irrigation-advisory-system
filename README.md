# AgriSat — AI Crop Monitoring & Irrigation Advisory
### Bharatiya Antariksh Hackathon 2026 | Challenge 6

> **Sentinel-2 Optical + Sentinel-1 SAR fusion for all-weather, stage-aware crop monitoring and irrigation advisory.**

---

## Project Structure

```
agrisat/
├── app.py                       ← Main Streamlit app
├── utils/
│   ├── __init__.py
│   ├── sentinel.py              ← Sentinel-1 SAR + Sentinel-2 optical (CDSE Statistics API)
│   ├── classifier.py            ← Crop type + growth stage + moisture stress classifier
│   ├── weather.py                ← OpenWeatherMap 7-day forecast
│   ├── soil.py                   ← SoilGrids ISRIC soil classification
│   └── advisory.py               ← OpenRouter LLM irrigation advisory
├── .streamlit/
│   ├── config.toml              ← Dark theme config
│   └── secrets.toml.example     ← Template for API keys (copy -> secrets.toml)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/agrisat-bah2026.git
cd agrisat-bah2026

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and fill in your keys

streamlit run app.py
```

---

## API Keys

All keys are optional — the app falls back to deterministic synthetic /
regional-default data if a key is missing, so it always runs end-to-end.

| Service | Purpose | How to get it |
|---|---|---|
| **Copernicus Data Space Ecosystem (CDSE)** | Real Sentinel-1 SAR + Sentinel-2 optical bands (NDVI, NDWI, Moisture Index, VV/VH) | Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu) → **User Settings → OAuth clients → Create** → copy `Client ID` and `Client Secret` |
| **OpenWeatherMap** | 7-day weather forecast | [openweathermap.org/api](https://openweathermap.org/api) — free, 1000 calls/day |
| **OpenRouter** | LLM-enhanced irrigation advisory text | [openrouter.ai](https://openrouter.ai) — free models available |
| **SoilGrids ISRIC** | Soil classification | No key needed — open REST API |

In `.streamlit/secrets.toml`:

```toml
CDSE_CLIENT_ID     = "your-client-id"
CDSE_CLIENT_SECRET = "your-client-secret"
OPENWEATHER_KEY    = "your_key_here"
OPENROUTER_KEY     = "sk-or-your_key_here"
```

---

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (`.streamlit/secrets.toml` is gitignored — never commit it)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Go to **Advanced settings → Secrets** and paste:

```toml
CDSE_CLIENT_ID     = "your-client-id"
CDSE_CLIENT_SECRET = "your-client-secret"
OPENWEATHER_KEY    = "your_key_here"
OPENROUTER_KEY     = "sk-or-your_key_here"
```

5. Click **Deploy** 

---

## Technical Architecture

```
Field Coordinates (lat, lon)
         │
         ▼
┌─────────────────────────────────────┐
│   Copernicus Data Space (CDSE)       │
│   Sentinel Hub Statistics API        │
│   Sentinel-2 L2A: NDVI, NDWI,        │
│     Moisture Index (B8A,B11), EVI    │
│   Sentinel-1 GRD: VV/VH backscatter  │
└────────────┬──────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│   Multi-Output Classifier            │
│   Branch A: Optical features         │
│   Branch B: SAR moisture features    │
│   → Crop Type (5 classes)            │
│   → Growth Stage (4 stages)          │
│   → Moisture Stress (3 levels)       │
└────────────┬──────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
OpenWeatherMap    SoilGrids ISRIC
7-day forecast    Soil classification
    │                 │
    └────────┬────────┘
             ▼
┌─────────────────────────────────────┐
│   Advisory Engine                    │
│   Rule-based logic                   │
│   + OpenRouter LLM enhancement       │
│   → Stage-aware recommendation       │
└────────────┬──────────────────────────┘
             │
             ▼
    Streamlit Dashboard
```

### Real data fetch (sentinel.py)

- Authenticates against CDSE via OAuth client-credentials grant
- Calls the Sentinel Hub **Statistics API** with a ~500m×500m polygon around
  the field coordinates — returns band-mean statistics directly as JSON
  (no GeoTIFF download required)
- **Sentinel-2 L2A**: NDVI `(B08-B04)/(B08+B04)`, NDWI `(B03-B08)/(B03+B08)`,
  Moisture Index `(B8A-B11)/(B8A+B11)`, EVI — fetched as a 15-day interval
  time series over the last 6 months for stage/trend analysis
- **Sentinel-1 GRD**: VV/VH backscatter, converted from linear power to dB
- Falls back to a deterministic synthetic scene (seeded from lat/lon) if
  credentials are absent or any request fails — the app never breaks

---

## Supported Crops & Regions

| Crop | Region | Growth Stages |
|---|---|---|
| Rice | Telangana, Andhra Pradesh | Germination → Vegetative → Flowering → Harvest |
| Wheat | Punjab, Haryana | Germination → Vegetative → Flowering → Harvest |
| Cotton | Maharashtra, Gujarat | Germination → Vegetative → Flowering → Harvest |
| Maize | Karnataka, Tamil Nadu | Germination → Vegetative → Flowering → Harvest |
| Soybean | MP, Maharashtra | Germination → Vegetative → Flowering → Harvest |

---

## Hackathon Notes

- **Challenge 6:** AI-Driven Automated Crop Type, Moisture Stress Detection
  and Irrigation Advisory Across Growth Stages Using Moderate Resolution
  Spectral Signatures (Optical & Microwave)
- **USP:** SAR + Optical fusion → cloud-penetrating, all-weather monitoring
- **Moisture Index** (B8A/B11 SWIR-based) used alongside NDWI and SAR VV for
  a stronger moisture-stress signal than NDVI/NDWI alone
- **Stage-aware advisory:** recommendations calibrated to germination /
  vegetative / flowering / harvest-ready phases
- **Tech stack:** Streamlit · Python · Copernicus Data Space Ecosystem ·
  OpenRouter · OpenWeatherMap · SoilGrids ISRIC

---

## License

MIT License — open for academic and non-commercial use.

Data sources: ESA Copernicus (Sentinel-1/2 via CDSE) · ISRIC SoilGrids · OpenWeatherMap
