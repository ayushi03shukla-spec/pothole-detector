import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import warnings
import os
import random
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PotholeAI — Road Health Scanner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"  # collapsed by default = better on mobile
)

# ─── Mobile-First CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ── CSS Variables ── */
:root {
    --orange: #ff6b00;
    --orange-light: #ffaa00;
    --orange-glow: rgba(255,107,0,0.15);
    --red: #ff2d2d;
    --green: #00e676;
    --bg: #0d0d0d;
    --surface: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.08);
    --text: #e8e0d0;
    --muted: #888;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a0a00 0%, #0d0d0d 50%, #001a0a 100%);
    min-height: 100vh;
}

/* ── Hero ── */
.hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(2rem, 10vw, 4.5rem);
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff6b00 0%, #ffaa00 50%, #ff6b00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3s ease-in-out infinite;
    background-size: 200% auto;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.hero-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.65rem, 2.5vw, 0.85rem);
    color: var(--muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}
.hero-divider {
    width: 50px; height: 3px;
    background: linear-gradient(90deg, #ff6b00, #ffaa00);
    margin: 1.2rem auto; border-radius: 2px;
}

/* ── Stat Pills — wrap nicely on mobile ── */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin: 0.8rem 0 1.5rem;
    padding: 0 0.5rem;
}
.stat-pill {
    background: rgba(255,107,0,0.08);
    border: 1px solid rgba(255,107,0,0.25);
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.65rem, 2vw, 0.75rem);
    color: #ff9944;
    letter-spacing: 0.03em;
    white-space: nowrap;
}

/* ── Section Heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.1rem, 4vw, 1.4rem);
    color: var(--text);
    margin: 1.5rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(255,107,0,0.2);
}

/* ── Location Box ── */
.location-box {
    background: rgba(0,150,255,0.06);
    border: 1px solid rgba(0,150,255,0.2);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: clamp(0.8rem, 3vw, 1rem);
}
.location-label {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.65rem, 2vw, 0.75rem);
    color: #4fc3f7;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ── Result Cards — full width on mobile ── */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: clamp(0.8rem, 3vw, 1.2rem);
    margin: 0.5rem 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    width: 100%;
    box-sizing: border-box;
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255,107,0,0.1);
}

/* ── Badges — larger on mobile ── */
.badge-danger {
    display: inline-block;
    background: linear-gradient(135deg, #ff2d2d, #ff6b00);
    color: white;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.7rem, 2.5vw, 0.78rem);
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.4rem 1.1rem;
    border-radius: 100px;
    text-transform: uppercase;
    box-shadow: 0 2px 12px rgba(255,45,45,0.35);
}
.badge-safe {
    display: inline-block;
    background: linear-gradient(135deg, #00c853, #00e676);
    color: #001a05;
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.7rem, 2.5vw, 0.78rem);
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.4rem 1.1rem;
    border-radius: 100px;
    text-transform: uppercase;
    box-shadow: 0 2px 12px rgba(0,200,83,0.35);
}

/* ── Severity ── */
.sev-low    { color: #00e676; font-family: 'Space Mono', monospace; font-size: clamp(0.75rem, 2.5vw, 0.8rem); }
.sev-medium { color: #ffaa00; font-family: 'Space Mono', monospace; font-size: clamp(0.75rem, 2.5vw, 0.8rem); }
.sev-high   { color: #ff2d2d; font-family: 'Space Mono', monospace; font-size: clamp(0.75rem, 2.5vw, 0.8rem); }

/* ── Confidence Bar ── */
.conf-bar-bg {
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    height: 10px;   /* slightly taller for touch screens */
    margin: 0.5rem 0;
    overflow: hidden;
}
.conf-bar-fill-danger {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #ff6b00, #ff2d2d);
}
.conf-bar-fill-safe {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #00c853, #00e676);
}

/* ── Filename ── */
.filename-label {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.65rem, 2vw, 0.72rem);
    color: #666;
    margin-bottom: 0.4rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

/* ── Summary Box — stack on mobile ── */
.summary-box {
    background: rgba(255,107,0,0.06);
    border: 1px solid rgba(255,107,0,0.2);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin: 1.2rem 0;
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
}
.summary-stat { text-align: center; min-width: 70px; }
.summary-stat .number {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.8rem, 6vw, 2.5rem);
    color: #ff8c00;
    line-height: 1;
}
.summary-stat .label {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.6rem, 1.8vw, 0.7rem);
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

/* ── Map heading ── */
.map-heading {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: clamp(1.1rem, 4vw, 1.4rem);
    color: var(--text);
    margin: 1.5rem 0 0.5rem;
}

/* ── Mobile: full width columns override ── */
@media (max-width: 768px) {
    /* Force single column on small screens */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    .result-card { margin: 0.4rem 0; }
    .summary-box { gap: 1rem; }
    /* Bigger upload button on mobile */
    [data-testid="stFileUploader"] label {
        font-size: 1rem !important;
        padding: 1.5rem !important;
    }
    /* Bigger inputs */
    input[type="text"] { font-size: 16px !important; } /* prevents zoom on iOS */
}

/* ── Streamlit overrides ── */
section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid rgba(255,107,0,0.15);
}
section[data-testid="stSidebar"] * { color: #c8c0b0 !important; }

[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,107,0,0.3) !important;
    border-radius: 16px !important;
    background: rgba(255,107,0,0.03) !important;
    padding: 1rem !important;
}

/* ── Download button — full width on mobile ── */
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff6b00, #ff8c00) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: clamp(0.75rem, 2.5vw, 0.8rem) !important;
    letter-spacing: 0.08em !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(255,107,0,0.4) !important;
}

/* ── Streamlit default text inputs — bigger on mobile ── */
.stTextInput input {
    font-size: clamp(0.9rem, 3vw, 1rem) !important;
    padding: 0.7rem 1rem !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,107,0,0.3) !important;
    color: var(--text) !important;
}
.stTextInput input:focus {
    border-color: var(--orange) !important;
    box-shadow: 0 0 0 2px rgba(255,107,0,0.2) !important;
}

/* ── Number inputs ── */
.stNumberInput input {
    font-size: 16px !important; /* prevents zoom on iOS */
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #ff6b00; border-radius: 3px; }

/* ── Tap highlight removal for mobile ── */
* { -webkit-tap-highlight-color: transparent; }
</style>
""", unsafe_allow_html=True)


# ─── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-title">POTHOLE AI</div>
    <div class="hero-subtitle">Road Health Intelligence System</div>
    <div class="hero-divider"></div>
    <div class="stats-row">
        <div class="stat-pill">⚡ CNN Powered</div>
        <div class="stat-pill">🎯 Severity Scoring</div>
        <div class="stat-pill">🗺️ GPS Map</div>
        <div class="stat-pill">📱 Mobile Ready</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛣️ PotholeAI")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Enter a road location
2. Upload road images
3. CNN scans each image
4. Pins appear on the map
5. Download CSV report
    """)
    st.markdown("---")
    st.markdown("**Map Legend**")
    st.markdown("""
- 🔴 High severity
- 🟠 Medium severity
- 🟡 Low severity
- 🟢 Normal road
    """)
    st.caption("TensorFlow · Streamlit · Folium")


# ─── Session State ─────────────────────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model("pothole_cnn_model.h5")

model = load_cnn_model()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_severity(confidence, is_pothole):
    if not is_pothole:
        return "Safe", "sev-low", "green"
    if confidence < 0.65:
        return "Low", "sev-low", "#f5c518"
    elif confidence < 0.85:
        return "Medium", "sev-medium", "orange"
    else:
        return "High", "sev-high", "red"

def geocode_location(location_str):
    try:
        geolocator = Nominatim(user_agent="potholeai_app")
        loc = geolocator.geocode(location_str, timeout=5)
        if loc:
            return loc.latitude, loc.longitude
        return None, None
    except Exception:
        return None, None

def marker_icon(severity_color):
    return {"red": "red", "orange": "orange", "#f5c518": "beige", "green": "green"}.get(severity_color, "blue")


# ─── Detect mobile screen hint ─────────────────────────────────────────────────
# Use 1 column on mobile by checking viewport via JS
st.markdown("""
<script>
    const isMobile = window.innerWidth < 768;
    window.parent.postMessage({type: 'mobile', value: isMobile}, '*');
</script>
""", unsafe_allow_html=True)


# ─── Location Input ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📍 Set Scan Location</div>', unsafe_allow_html=True)

use_coords = st.checkbox("📌 Use coordinates instead of location name")

lat, lon = None, None

if use_coords:
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=77.2090, format="%.6f")
    location_name = f"{lat:.4f}, {lon:.4f}"
    st.markdown(f'<div class="location-box"><div class="location-label">📌 Coordinates set</div><b>{lat}, {lon}</b></div>', unsafe_allow_html=True)
else:
    location_input = st.text_input(
        "Road name",
        placeholder="e.g. MG Road Bangalore, Connaught Place New Delhi",
        label_visibility="collapsed"
    )
    if location_input:
        with st.spinner("📡 Finding location..."):
            lat, lon = geocode_location(location_input)
        if lat and lon:
            location_name = location_input
            st.markdown(f'<div class="location-box"><div class="location-label">✅ Location found</div><b>{location_input}</b><br><span style="color:#888; font-size:0.8rem;">{lat:.5f}, {lon:.5f}</span></div>', unsafe_allow_html=True)
        else:
            st.error("❌ Location not found. Try adding city name e.g. 'MG Road, Bangalore, India'")
            location_name = "Unknown"
    else:
        location_name = "Unknown"


# ─── File Uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📂 Upload Road Images</div>', unsafe_allow_html=True)
st.caption("📱 On mobile: tap the box below to open your camera or gallery")

uploaded_files = st.file_uploader(
    "Tap to upload or take photo",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

results = []

# ─── Prediction Logic ──────────────────────────────────────────────────────────
if uploaded_files:
    if not lat or not lon:
        st.warning("⚠️ No location set — set a location above to see pins on the map!")

    st.markdown(f'<div class="section-heading">🔍 Results — {len(uploaded_files)} image(s)</div>', unsafe_allow_html=True)

    # Responsive columns: 1 on mobile, up to 3 on desktop
    # We use max 2 columns to keep cards readable on all screens
    num_cols = min(len(uploaded_files), 2)
    rows = [uploaded_files[i:i + num_cols] for i in range(0, len(uploaded_files), num_cols)]

    new_scan_points = []

    for row in rows:
        cols = st.columns(len(row))
        for idx, uploaded_file in enumerate(row):
            with cols[idx]:
                st.image(uploaded_file, use_column_width=True)

                img = image.load_img(uploaded_file, target_size=(128, 128))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = model.predict(img_array, verbose=0)
                raw_conf = float(prediction[0][0])

                is_pothole = raw_conf > 0.5
                conf_score = raw_conf if is_pothole else 1 - raw_conf
                severity_label, severity_css, severity_color = get_severity(conf_score, is_pothole)

                badge = '<span class="badge-danger">⚠ Pothole</span>' if is_pothole else '<span class="badge-safe">✓ Normal</span>'
                bar_class = "conf-bar-fill-danger" if is_pothole else "conf-bar-fill-safe"
                bar_pct = int(conf_score * 100)
                sev_html = f'<span class="{severity_css}">{severity_label}</span>'

                st.markdown(f"""
<div class="result-card">
    <div class="filename-label">📄 {uploaded_file.name}</div>
    {badge}
    <div style="margin-top:0.8rem;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#888;">CONFIDENCE</span>
            <span style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#ccc;">{bar_pct}%</span>
        </div>
        <div class="conf-bar-bg"><div class="{bar_class}" style="width:{bar_pct}%;"></div></div>
    </div>
    <div style="margin-top:0.6rem; font-family:'Space Mono',monospace; font-size:0.75rem; color:#888;">
        SEVERITY &nbsp; {sev_html}
    </div>
</div>
""", unsafe_allow_html=True)

                prediction_text = "Pothole" if is_pothole else "Normal Road"
                results.append({
                    "Filename": uploaded_file.name,
                    "Prediction": prediction_text,
                    "Confidence (%)": f"{bar_pct}%",
                    "Severity": severity_label,
                    "Location": location_name,
                    "Latitude": lat if lat else "N/A",
                    "Longitude": lon if lon else "N/A",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

                if lat and lon:
                    offset = 0.0003
                    new_scan_points.append({
                        "lat": lat + random.uniform(-offset, offset),
                        "lon": lon + random.uniform(-offset, offset),
                        "filename": uploaded_file.name,
                        "prediction": prediction_text,
                        "confidence": bar_pct,
                        "severity": severity_label,
                        "severity_color": severity_color,
                        "location": location_name,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

    if new_scan_points:
        st.session_state.scan_history.extend(new_scan_points)

    # ─── Summary ───────────────────────────────────────────────────────────────
    if results:
        total = len(results)
        potholes = sum(1 for r in results if r["Prediction"] == "Pothole")
        normal = total - potholes
        avg_conf = sum(int(r["Confidence (%)"].replace("%", "")) for r in results) / total

        st.markdown(f"""
<div class="summary-box">
    <div class="summary-stat">
        <div class="number">{total}</div>
        <div class="label">Scanned</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#ff2d2d">{potholes}</div>
        <div class="label">Potholes</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#00e676">{normal}</div>
        <div class="label">Normal</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#ffaa00">{avg_conf:.0f}%</div>
        <div class="label">Avg Conf</div>
    </div>
</div>
""", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Report (CSV)",
            data=csv,
            file_name='pothole_report.csv',
            mime='text/csv'
        )


# ─── Map Section ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="map-heading">🗺️ Live Pothole Map</div>', unsafe_allow_html=True)
st.caption("Pins persist across uploads in this session. Tap a pin to see details.")

col_btn, col_metric = st.columns([2, 1])
with col_btn:
    if st.button("🗑️ Clear All Pins"):
        st.session_state.scan_history = []
        st.rerun()
with col_metric:
    pothole_pins = sum(1 for p in st.session_state.scan_history if p["prediction"] == "Pothole")
    st.metric("🔴 Potholes on map", pothole_pins)

# Map center
if st.session_state.scan_history:
    center_lat = st.session_state.scan_history[-1]["lat"]
    center_lon = st.session_state.scan_history[-1]["lon"]
elif lat and lon:
    center_lat, center_lon = lat, lon
else:
    center_lat, center_lon = 20.5937, 78.9629

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=14,
    tiles="CartoDB dark_matter"
)

for point in st.session_state.scan_history:
    icon_color = marker_icon(point["severity_color"])
    emoji = "⚠️" if point["prediction"] == "Pothole" else "✅"
    popup_html = f"""
    <div style="font-family:monospace; min-width:160px; font-size:13px;">
        <b>{emoji} {point['prediction']}</b><br>
        📄 {point['filename']}<br>
        📊 {point['confidence']}% confidence<br>
        🔆 {point['severity']} severity<br>
        📍 {point['location']}<br>
        🕐 {point['timestamp']}
    </div>
    """
    folium.Marker(
        location=[point["lat"], point["lon"]],
        popup=folium.Popup(popup_html, max_width=200),
        tooltip=f"{emoji} {point['prediction']} ({point['confidence']}%)",
        icon=folium.Icon(
            color=icon_color,
            icon="exclamation-sign" if point["prediction"] == "Pothole" else "ok-sign",
            prefix="glyphicon"
        )
    ).add_to(m)

if not st.session_state.scan_history and lat and lon:
    folium.Marker(
        location=[lat, lon],
        tooltip="📍 Selected location",
        icon=folium.Icon(color="blue", icon="map-marker")
    ).add_to(m)

# Map height: smaller on mobile, larger on desktop
st_folium(m, width="100%", height=400)


# ─── Session History Table ─────────────────────────────────────────────────────
if st.session_state.scan_history:
    st.markdown("---")
    st.markdown('<div class="section-heading">📋 Session History</div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.scan_history)[
        ["filename", "prediction", "confidence", "severity", "location", "timestamp"]
    ]
    history_df.columns = ["File", "Prediction", "Confidence (%)", "Severity", "Location", "Time"]
    st.dataframe(history_df, use_container_width=True, hide_index=True)

    full_csv = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Full Session History (CSV)",
        data=full_csv,
        file_name="pothole_session_history.csv",
        mime="text/csv"
    )