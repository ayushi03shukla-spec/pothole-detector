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
import json
from datetime import datetime
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PotholeAI — Road Health Scanner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0d0d0d;
    color: #e8e0d0;
}
.stApp {
    background: radial-gradient(ellipse at 20% 50%, #1a0a00 0%, #0d0d0d 50%, #001a0a 100%);
    min-height: 100vh;
}
.hero-header { text-align: center; padding: 3rem 1rem 2rem; }
.hero-title {
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: clamp(2.5rem, 6vw, 4.5rem); letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff6b00 0%, #ffaa00 50%, #ff6b00 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; animation: shimmer 3s ease-in-out infinite; background-size: 200% auto;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    50% { background-position: 100% center; }
    100% { background-position: 0% center; }
}
.hero-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.85rem;
    color: #888; letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.5rem;
}
.hero-divider { width: 60px; height: 3px; background: linear-gradient(90deg, #ff6b00, #ffaa00); margin: 1.5rem auto; border-radius: 2px; }
.stats-row { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin: 1rem 0 2rem; }
.stat-pill {
    background: rgba(255,107,0,0.08); border: 1px solid rgba(255,107,0,0.25);
    border-radius: 100px; padding: 0.4rem 1.2rem;
    font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #ff9944; letter-spacing: 0.05em;
}
.result-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.2rem; margin: 0.5rem 0;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.result-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(255,107,0,0.1); }
.badge-danger {
    display: inline-block; background: linear-gradient(135deg, #ff2d2d, #ff6b00);
    color: white; font-family: 'Space Mono', monospace; font-size: 0.78rem;
    font-weight: 700; letter-spacing: 0.08em; padding: 0.35rem 1rem;
    border-radius: 100px; text-transform: uppercase; box-shadow: 0 2px 12px rgba(255,45,45,0.35);
}
.badge-safe {
    display: inline-block; background: linear-gradient(135deg, #00c853, #00e676);
    color: #001a05; font-family: 'Space Mono', monospace; font-size: 0.78rem;
    font-weight: 700; letter-spacing: 0.08em; padding: 0.35rem 1rem;
    border-radius: 100px; text-transform: uppercase; box-shadow: 0 2px 12px rgba(0,200,83,0.35);
}
.sev-low    { color: #00e676; font-family: 'Space Mono', monospace; font-size:0.8rem; }
.sev-medium { color: #ffaa00; font-family: 'Space Mono', monospace; font-size:0.8rem; }
.sev-high   { color: #ff2d2d; font-family: 'Space Mono', monospace; font-size:0.8rem; }
.conf-bar-bg { background: rgba(255,255,255,0.07); border-radius: 100px; height: 8px; margin: 0.5rem 0; overflow: hidden; }
.conf-bar-fill-danger { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #ff6b00, #ff2d2d); }
.conf-bar-fill-safe   { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #00c853, #00e676); }
.filename-label { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #666; margin-bottom: 0.3rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.section-heading { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem; color: #e8e0d0; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,107,0,0.2); }
.summary-box { background: rgba(255,107,0,0.06); border: 1px solid rgba(255,107,0,0.2); border-radius: 16px; padding: 1.5rem 2rem; margin: 1.5rem 0; display: flex; gap: 3rem; flex-wrap: wrap; }
.summary-stat { text-align: center; }
.summary-stat .number { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.5rem; color: #ff8c00; line-height: 1; }
.summary-stat .label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }
.location-box { background: rgba(0,150,255,0.06); border: 1px solid rgba(0,150,255,0.2); border-radius: 16px; padding: 1.2rem 1.5rem; margin: 1rem 0; }
.location-label { font-family: 'Space Mono', monospace; font-size: 0.75rem; color: #4fc3f7; letter-spacing: 0.1em; text-transform: uppercase; }
.map-heading { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem; color: #e8e0d0; margin: 2rem 0 0.5rem; }
section[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid rgba(255,107,0,0.15); }
section[data-testid="stSidebar"] * { color: #c8c0b0 !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg, #ff6b00, #ff8c00) !important;
    color: #000 !important; font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important; font-size: 0.8rem !important;
    letter-spacing: 0.08em !important; border: none !important;
    border-radius: 100px !important; padding: 0.6rem 2rem !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d0d0d; }
::-webkit-scrollbar-thumb { background: #ff6b00; border-radius: 3px; }
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
        <div class="stat-pill">🎯 Binary Classification</div>
        <div class="stat-pill">📊 Severity Scoring</div>
        <div class="stat-pill">🗺️ GPS Map Pinning</div>
        <div class="stat-pill">📥 CSV Export</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛣️ PotholeAI")
    st.markdown("---")
    st.markdown("""
**How it works:**
1. Enter a road location name or coordinates
2. Upload road images (JPG/PNG)
3. CNN model scans each image
4. Potholes are pinned on the live map
5. Download full report as CSV
    """)
    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("""
- Architecture: CNN (Keras)
- Input size: 128×128 px
- Output: Binary + Confidence
- Severity: 3-tier scale
    """)
    st.markdown("---")
    st.markdown("**Map Legend**")
    st.markdown("""
- 🔴 High severity pothole
- 🟠 Medium severity pothole
- 🟡 Low severity pothole
- 🟢 Normal road (safe)
    """)
    st.caption("Built with TensorFlow · Streamlit · Folium")


# ─── Session State for map history ─────────────────────────────────────────────
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
    """Convert location string to (lat, lon)."""
    try:
        geolocator = Nominatim(user_agent="potholeai_app")
        loc = geolocator.geocode(location_str, timeout=5)
        if loc:
            return loc.latitude, loc.longitude
        return None, None
    except Exception:
        return None, None

def marker_icon(severity_color):
    color_map = {"red": "red", "orange": "orange", "#f5c518": "beige", "green": "green"}
    return color_map.get(severity_color, "blue")


# ─── Location Input ────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📍 Set Scan Location</div>', unsafe_allow_html=True)

col_loc1, col_loc2 = st.columns([3, 1])

with col_loc1:
    location_input = st.text_input(
        "Enter road/area name (e.g. MG Road Bangalore, NH-44 Delhi)",
        placeholder="e.g. Connaught Place, New Delhi",
        label_visibility="collapsed"
    )

with col_loc2:
    use_coords = st.checkbox("Use coordinates instead")

lat, lon = None, None

if use_coords:
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=28.6139, format="%.6f")
    with c2:
        lon = st.number_input("Longitude", value=77.2090, format="%.6f")
    location_name = f"{lat:.4f}, {lon:.4f}"
    st.markdown(f'<div class="location-box"><div class="location-label">📌 Coordinates set</div><b>{lat}, {lon}</b></div>', unsafe_allow_html=True)

elif location_input:
    with st.spinner("📡 Geocoding location..."):
        lat, lon = geocode_location(location_input)
    if lat and lon:
        location_name = location_input
        st.markdown(f'<div class="location-box"><div class="location-label">📌 Location resolved</div><b>{location_input}</b> → {lat:.5f}, {lon:.5f}</div>', unsafe_allow_html=True)
    else:
        st.error("❌ Could not find that location. Try a more specific name or use coordinates.")
        location_name = "Unknown"
else:
    location_name = "Unknown"


# ─── File Uploader ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📂 Upload Road Images</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drag & drop images here or click to select",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

results = []

# ─── Prediction Logic ──────────────────────────────────────────────────────────
if uploaded_files:
    if not lat or not lon:
        st.warning("⚠️ No location set — predictions will run but won't appear on the map. Set a location above!")

    st.markdown(f'<div class="section-heading">🔍 Scan Results — {len(uploaded_files)} image(s)</div>', unsafe_allow_html=True)

    num_cols = min(len(uploaded_files), 3)
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

                badge = '<span class="badge-danger">⚠ Pothole Detected</span>' if is_pothole else '<span class="badge-safe">✓ Normal Road</span>'
                bar_class = "conf-bar-fill-danger" if is_pothole else "conf-bar-fill-safe"
                bar_pct = int(conf_score * 100)

                sev_display = f'<span class="{severity_css}">{severity_label}</span>'

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
        SEVERITY &nbsp; {sev_display}
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

                # Add to map points if location is set
                if lat and lon:
                    # Add small random offset so pins don't overlap perfectly
                    import random
                    offset = 0.0003
                    jitter_lat = lat + random.uniform(-offset, offset)
                    jitter_lon = lon + random.uniform(-offset, offset)

                    new_scan_points.append({
                        "lat": jitter_lat,
                        "lon": jitter_lon,
                        "filename": uploaded_file.name,
                        "prediction": prediction_text,
                        "confidence": bar_pct,
                        "severity": severity_label,
                        "severity_color": severity_color,
                        "location": location_name,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })

    # Save to session history
    if new_scan_points:
        st.session_state.scan_history.extend(new_scan_points)

    # ─── Summary Stats ─────────────────────────────────────────────────────────
    if results:
        total = len(results)
        potholes = sum(1 for r in results if r["Prediction"] == "Pothole")
        normal = total - potholes
        avg_conf = sum(int(r["Confidence (%)"].replace("%", "")) for r in results) / total

        st.markdown(f"""
<div class="summary-box">
    <div class="summary-stat">
        <div class="number">{total}</div>
        <div class="label">Images Scanned</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#ff2d2d">{potholes}</div>
        <div class="label">Potholes Found</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#00e676">{normal}</div>
        <div class="label">Normal Roads</div>
    </div>
    <div class="summary-stat">
        <div class="number" style="color:#ffaa00">{avg_conf:.0f}%</div>
        <div class="label">Avg Confidence</div>
    </div>
</div>
""", unsafe_allow_html=True)

        df = pd.DataFrame(results)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv,
            file_name='pothole_scan_report.csv',
            mime='text/csv'
        )


# ─── Interactive Map ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="map-heading">🗺️ Live Pothole Map</div>', unsafe_allow_html=True)
st.caption("All scanned locations are pinned below. Pins persist across uploads in this session.")

col_map1, col_map2 = st.columns([4, 1])

with col_map2:
    if st.button("🗑️ Clear Map History"):
        st.session_state.scan_history = []
        st.rerun()
    st.markdown("**Total pins:**")
    st.metric("", len(st.session_state.scan_history))
    pothole_pins = sum(1 for p in st.session_state.scan_history if p["prediction"] == "Pothole")
    st.metric("🔴 Potholes", pothole_pins)
    st.metric("🟢 Normal", len(st.session_state.scan_history) - pothole_pins)

with col_map1:
    # Determine map center
    if st.session_state.scan_history:
        center_lat = st.session_state.scan_history[-1]["lat"]
        center_lon = st.session_state.scan_history[-1]["lon"]
    elif lat and lon:
        center_lat, center_lon = lat, lon
    else:
        center_lat, center_lon = 20.5937, 78.9629  # Default: India center

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB dark_matter"
    )

    # Add all history pins
    for point in st.session_state.scan_history:
        icon_color = marker_icon(point["severity_color"])
        emoji = "⚠️" if point["prediction"] == "Pothole" else "✅"

        popup_html = f"""
        <div style="font-family:monospace; min-width:180px;">
            <b>{emoji} {point['prediction']}</b><br>
            📄 {point['filename']}<br>
            📊 Confidence: {point['confidence']}%<br>
            🔆 Severity: {point['severity']}<br>
            📍 {point['location']}<br>
            🕐 {point['timestamp']}
        </div>
        """

        folium.Marker(
            location=[point["lat"], point["lon"]],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{emoji} {point['prediction']} ({point['confidence']}%)",
            icon=folium.Icon(color=icon_color, icon="exclamation-sign" if point["prediction"] == "Pothole" else "ok-sign", prefix="glyphicon")
        ).add_to(m)

    # If no history but location is set, show location marker
    if not st.session_state.scan_history and lat and lon:
        folium.Marker(
            location=[lat, lon],
            tooltip="📍 Selected location (upload images to scan)",
            icon=folium.Icon(color="blue", icon="map-marker")
        ).add_to(m)

    st_folium(m, width="100%", height=480)


# ─── Session History Table ─────────────────────────────────────────────────────
if st.session_state.scan_history:
    st.markdown("---")
    st.markdown('<div class="section-heading">📋 Session Scan History</div>', unsafe_allow_html=True)
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
