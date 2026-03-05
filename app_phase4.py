try:
    from firebase_client import save_detection, fetch_detections
    FIREBASE_AVAILABLE = True
except Exception:
    FIREBASE_AVAILABLE = False
    def save_detection(x): pass
    def fetch_detections(**kwargs): return []
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
import warnings
import os
import random
import smtplib
import hashlib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PotholeAI — Road Health Scanner",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --orange: #ff6b00; --orange-light: #ffaa00;
    --red: #ff2d2d; --green: #00e676;
    --bg: #0d0d0d; --surface: rgba(255,255,255,0.03);
    --border: rgba(255,255,255,0.08); --text: #e8e0d0; --muted: #888;
}
html, body, [class*="css"] { font-family: 'Syne', sans-serif; background-color: var(--bg); color: var(--text); }
.stApp { background: radial-gradient(ellipse at 20% 50%, #1a0a00 0%, #0d0d0d 50%, #001a0a 100%); min-height: 100vh; }
.hero-header { text-align:center; padding: 2rem 1rem 1.5rem; }
.hero-title {
    font-weight: 800; font-size: clamp(2rem, 10vw, 4.5rem); letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ff6b00 0%, #ffaa00 50%, #ff6b00 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    animation: shimmer 3s ease-in-out infinite; background-size: 200% auto;
}
@keyframes shimmer { 0%,100% { background-position: 0% center; } 50% { background-position: 100% center; } }
.hero-subtitle { font-family:'Space Mono',monospace; font-size:clamp(0.65rem,2.5vw,0.85rem); color:var(--muted); letter-spacing:0.15em; text-transform:uppercase; margin-top:0.4rem; }
.hero-divider { width:50px; height:3px; background:linear-gradient(90deg,#ff6b00,#ffaa00); margin:1.2rem auto; border-radius:2px; }
.stats-row { display:flex; justify-content:center; gap:0.5rem; flex-wrap:wrap; margin:0.8rem 0 1.5rem; padding:0 0.5rem; }
.stat-pill { background:rgba(255,107,0,0.08); border:1px solid rgba(255,107,0,0.25); border-radius:100px; padding:0.3rem 0.9rem; font-family:'Space Mono',monospace; font-size:clamp(0.65rem,2vw,0.75rem); color:#ff9944; white-space:nowrap; }
.section-heading { font-weight:800; font-size:clamp(1.1rem,4vw,1.4rem); margin:1.5rem 0 0.8rem; padding-bottom:0.5rem; border-bottom:1px solid rgba(255,107,0,0.2); }
.location-box { background:rgba(0,150,255,0.06); border:1px solid rgba(0,150,255,0.2); border-radius:12px; padding:1rem 1.2rem; margin:0.8rem 0; }
.location-label { font-family:'Space Mono',monospace; font-size:clamp(0.65rem,2vw,0.75rem); color:#4fc3f7; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem; }
.result-card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:clamp(0.8rem,3vw,1.2rem); margin:0.5rem 0; width:100%; box-sizing:border-box; }
.badge-danger { display:inline-block; background:linear-gradient(135deg,#ff2d2d,#ff6b00); color:#fff; font-family:'Space Mono',monospace; font-size:clamp(0.7rem,2.5vw,0.78rem); font-weight:700; padding:0.4rem 1.1rem; border-radius:100px; text-transform:uppercase; }
.badge-safe { display:inline-block; background:linear-gradient(135deg,#00c853,#00e676); color:#001a05; font-family:'Space Mono',monospace; font-size:clamp(0.7rem,2.5vw,0.78rem); font-weight:700; padding:0.4rem 1.1rem; border-radius:100px; text-transform:uppercase; }
.sev-low    { color:#00e676; font-family:'Space Mono',monospace; font-size:0.8rem; }
.sev-medium { color:#ffaa00; font-family:'Space Mono',monospace; font-size:0.8rem; }
.sev-high   { color:#ff2d2d; font-family:'Space Mono',monospace; font-size:0.8rem; }
.conf-bar-bg { background:rgba(255,255,255,0.07); border-radius:100px; height:10px; margin:0.5rem 0; overflow:hidden; }
.conf-bar-fill-danger { height:100%; border-radius:100px; background:linear-gradient(90deg,#ff6b00,#ff2d2d); }
.conf-bar-fill-safe   { height:100%; border-radius:100px; background:linear-gradient(90deg,#00c853,#00e676); }
.filename-label { font-family:'Space Mono',monospace; font-size:0.72rem; color:#666; margin-bottom:0.4rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.summary-box { background:rgba(255,107,0,0.06); border:1px solid rgba(255,107,0,0.2); border-radius:16px; padding:1.2rem 1.5rem; margin:1.2rem 0; display:flex; gap:1.5rem; flex-wrap:wrap; justify-content:center; }
.summary-stat { text-align:center; min-width:70px; }
.summary-stat .number { font-weight:800; font-size:clamp(1.8rem,6vw,2.5rem); color:#ff8c00; line-height:1; }
.summary-stat .label { font-family:'Space Mono',monospace; font-size:0.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; margin-top:0.3rem; }
.map-heading { font-weight:800; font-size:clamp(1.1rem,4vw,1.4rem); margin:1.5rem 0 0.5rem; }

/* ── Login Card ── */
.login-card { background:rgba(255,107,0,0.05); border:1px solid rgba(255,107,0,0.2); border-radius:20px; padding:2.5rem; max-width:420px; margin:2rem auto; text-align:center; }
.login-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.8rem; color:#ff8c00; margin-bottom:0.3rem; }
.login-subtitle { font-family:'Space Mono',monospace; font-size:0.75rem; color:#888; margin-bottom:1.5rem; letter-spacing:0.1em; text-transform:uppercase; }
.user-badge { display:inline-flex; align-items:center; gap:0.5rem; background:rgba(255,107,0,0.1); border:1px solid rgba(255,107,0,0.3); border-radius:100px; padding:0.4rem 1rem; font-family:'Space Mono',monospace; font-size:0.78rem; color:#ff9944; margin-bottom:1rem; }

/* ── Email Box ── */
.email-box { background:rgba(255,107,0,0.05); border:1px solid rgba(255,107,0,0.25); border-radius:16px; padding:1.5rem; margin:1.5rem 0; }
.email-title { font-family:'Syne',sans-serif; font-weight:800; font-size:1.1rem; color:#ff9944; margin-bottom:1rem; }
.email-success { background:rgba(0,200,83,0.08); border:1px solid rgba(0,200,83,0.3); border-radius:12px; padding:1rem 1.2rem; font-family:'Space Mono',monospace; font-size:0.85rem; color:#00e676; margin-top:1rem; }
.email-error { background:rgba(255,45,45,0.08); border:1px solid rgba(255,45,45,0.3); border-radius:12px; padding:1rem 1.2rem; font-family:'Space Mono',monospace; font-size:0.85rem; color:#ff6b6b; margin-top:1rem; }

@media (max-width:768px) {
    [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:100% !important; }
    input[type="text"] { font-size:16px !important; }
}
[data-testid="stFileUploader"] { border:2px dashed rgba(255,107,0,0.3) !important; border-radius:16px !important; background:rgba(255,107,0,0.03) !important; padding:1rem !important; }
.stDownloadButton > button { background:linear-gradient(135deg,#ff6b00,#ff8c00) !important; color:#000 !important; font-family:'Space Mono',monospace !important; font-weight:700 !important; border:none !important; border-radius:100px !important; padding:0.7rem 2rem !important; width:100% !important; }
section[data-testid="stSidebar"] { background:#0a0a0a; border-right:1px solid rgba(255,107,0,0.15); }
section[data-testid="stSidebar"] * { color:#c8c0b0 !important; }
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-thumb { background:#ff6b00; border-radius:3px; }
* { -webkit-tap-highlight-color:transparent; }
</style>
""", unsafe_allow_html=True)


# ─── Session State ─────────────────────────────────────────────────────────────
if "scan_history"  not in st.session_state: st.session_state.scan_history  = []
if "logged_in"     not in st.session_state: st.session_state.logged_in     = False
if "username"      not in st.session_state: st.session_state.username      = ""
if "scan_results"  not in st.session_state: st.session_state.scan_results  = []


# ─── User Store (SHA-256 hashed passwords) ────────────────────────────────────
USERS = {
    "admin":  hashlib.sha256("admin123".encode()).hexdigest(),
    "ayushi": hashlib.sha256("pothole2024".encode()).hexdigest(),
}

def hash_pw(pw):      return hashlib.sha256(pw.encode()).hexdigest()
def check_login(u,p): return USERS.get(u.lower()) == hash_pw(p)
def register(u,p):
    if u.lower() in USERS: return False, "Username already exists!"
    USERS[u.lower()] = hash_pw(p)
    return True, "Account created! Please login."


# ─── Google OAuth Setup (Direct URL method) ───────────────────────────────────
try:
    GOOGLE_CLIENT_ID     = st.secrets["google"]["client_id"]
    GOOGLE_CLIENT_SECRET = st.secrets["google"]["client_secret"]
    REDIRECT_URI         = st.secrets["google"]["redirect_uri"]
    GOOGLE_LOGIN_AVAILABLE = True
except Exception:
    GOOGLE_LOGIN_AVAILABLE = False

def get_google_auth_url():
    import urllib.parse
    params = {
        "client_id":     GOOGLE_CLIENT_ID,
        "redirect_uri":  REDIRECT_URI,
        "response_type": "code",
        "scope":         "openid email profile",
        "access_type":   "offline",
        "prompt":        "consent",
    }
    return "https://accounts.google.com/o/oauth2/auth?" + urllib.parse.urlencode(params)

def exchange_code_for_token(code):
    import requests
    data = {
        "code":          code,
        "client_id":     GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri":  REDIRECT_URI,
        "grant_type":    "authorization_code",
    }
    r = requests.post("https://oauth2.googleapis.com/token", data=data)
    return r.json()

def get_google_user_info(access_token):
    import requests
    r = requests.get("https://www.googleapis.com/oauth2/v1/userinfo",
                     params={"access_token": access_token})
    return r.json()


# ─── LOGIN PAGE ────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown("""
    <div class="hero-header">
        <div class="hero-title">POTHOLE AI</div>
        <div class="hero-subtitle">Road Health Intelligence System</div>
        <div class="hero-divider"></div>
    </div>""", unsafe_allow_html=True)

    _, col, _ = st.columns([1,2,1])
    with col:
        st.markdown('<div class="login-card">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Welcome</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Sign in to continue</div>', unsafe_allow_html=True)

        # ── Google Login Button ──
        if GOOGLE_LOGIN_AVAILABLE:

            # Check if returning from Google with ?code=...
            query_params = st.query_params
            if "code" in query_params:
                with st.spinner("🔄 Signing in with Google..."):
                    try:
                        code       = query_params["code"]
                        token_data = exchange_code_for_token(code)
                        user_info  = get_google_user_info(token_data["access_token"])
                        name       = user_info.get("name",  user_info.get("email","google_user"))
                        username   = name.replace(" ","_").lower()
                        st.session_state.logged_in    = True
                        st.session_state.username     = username
                        st.session_state.login_method = "google"
                        st.query_params.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Google login failed: {e}")
            else:
                google_url = get_google_auth_url()
                st.markdown(f"""
                <div style="margin-bottom:0.5rem;">
                    <a href="{google_url}" target="_self" style="text-decoration:none;">
                        <div style="
                            display:flex; align-items:center; justify-content:center; gap:12px;
                            background:#fff; color:#333; font-family:'Syne',sans-serif;
                            font-weight:700; font-size:0.95rem;
                            padding:0.7rem 1.5rem; border-radius:100px;
                            cursor:pointer; transition:all 0.2s ease;
                            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
                        ">
                            <img src="https://www.google.com/favicon.ico" width="20" height="20"/>
                            Continue with Google
                        </div>
                    </a>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("""
            <div style="display:flex; align-items:center; gap:0.8rem; margin:1rem 0;">
                <div style="flex:1; height:1px; background:rgba(255,255,255,0.1);"></div>
                <span style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#555;">OR</span>
                <div style="flex:1; height:1px; background:rgba(255,255,255,0.1);"></div>
            </div>
            """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            u = st.text_input("Username", placeholder="Enter username", key="lu")
            p = st.text_input("Password", placeholder="Enter password", type="password", key="lp")
            if st.button("🚀 Login", use_container_width=True):
                if check_login(u, p):
                    st.session_state.logged_in = True
                    st.session_state.username  = u
                    st.session_state.login_method = "password"
                    st.rerun()
                else:
                    st.error("❌ Wrong username or password!")
            st.markdown("---")
            st.caption("Demo → **admin** / **admin123**")

        with tab2:
            ru  = st.text_input("Choose Username", placeholder="e.g. ayushi", key="ru")
            rp  = st.text_input("Choose Password", placeholder="Min 6 characters", type="password", key="rp")
            rp2 = st.text_input("Confirm Password", placeholder="Repeat password",  type="password", key="rp2")
            if st.button("✅ Create Account", use_container_width=True):
                if not ru or not rp:
                    st.error("❌ Fill all fields!")
                elif len(rp) < 6:
                    st.error("❌ Password must be 6+ characters!")
                elif rp != rp2:
                    st.error("❌ Passwords don't match!")
                else:
                    ok, msg = register(ru, rp)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)

        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()


# ─── MAIN APP (only shown when logged in) ──────────────────────────────────────

st.markdown(f"""
<div class="hero-header">
    <div class="hero-title">POTHOLE AI</div>
    <div class="hero-subtitle">Road Health Intelligence System</div>
    <div class="hero-divider"></div>
    <div style="margin-bottom:0.5rem;"><span class="user-badge">👤 {st.session_state.username.upper()}</span></div>
    <div class="stats-row">
        <div class="stat-pill">⚡ CNN Powered</div>
        <div class="stat-pill">🎯 Severity Scoring</div>
        <div class="stat-pill">🗺️ GPS Map</div>
        <div class="stat-pill">📧 Email Reports</div>
        <div class="stat-pill">🔐 User Login</div>
        <div class="stat-pill">☁️ Firestore Logs</div>
        <div class="stat-pill">📊 Analytics</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username.upper()}")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in  = False
        st.session_state.username   = ""
        st.session_state.scan_history = []
        st.session_state.scan_results = []
        st.rerun()
    st.markdown("---")
    st.markdown("### 🛣️ PotholeAI")
    st.markdown("""
**How it works:**
1. Enter road location
2. Upload road images
3. CNN scans each image
4. Pins appear on map
5. Saved to Firestore
6. Send email report
7. Download CSV
    """)
    st.markdown("---")
    st.markdown("**📧 Email Settings**")
    sender_email    = st.text_input("Your Gmail",    placeholder="you@gmail.com")
    sender_password = st.text_input("App Password",  placeholder="xxxx xxxx xxxx xxxx", type="password")
    st.caption("Gmail → Security → App Passwords")
    st.markdown("---")
    st.markdown("**Map Legend**")
    st.markdown("🔴 High · 🟠 Medium · 🟡 Low · 🟢 Normal")
    st.markdown("---")
    st.markdown("### 🔎 Firestore Filters")
    flt_severity = st.multiselect("Severity",   ["High","Medium","Low","Safe"],  default=["High","Medium","Low","Safe"])
    flt_pred     = st.multiselect("Prediction", ["Pothole","Normal Road"],        default=["Pothole","Normal Road"])
    flt_limit    = st.slider("Max records", 50, 500, 200, step=50)
    st.caption("TensorFlow · Streamlit · Folium · Firestore")


# ─── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    return load_model("pothole_cnn_model.h5")
model = load_cnn_model()


# ─── Helpers ───────────────────────────────────────────────────────────────────
def get_severity(conf, is_pothole):
    if not is_pothole: return "Safe","sev-low","green"
    if conf < 0.65:    return "Low","sev-low","#f5c518"
    elif conf < 0.85:  return "Medium","sev-medium","orange"
    else:              return "High","sev-high","red"

def geocode_location(s):
    try:
        g = Nominatim(user_agent="potholeai_app")
        loc = g.geocode(s, timeout=5)
        return (loc.latitude, loc.longitude) if loc else (None,None)
    except: return None,None

def marker_icon(c):
    return {"red":"red","orange":"orange","#f5c518":"beige","green":"green"}.get(c,"blue")

def safe_fetch_detections(n=200):
    try:    return fetch_detections(limit=n)
    except TypeError:
        d = fetch_detections()
        return d[:n] if isinstance(d,list) else d
    except Exception as e:
        st.warning(f"Firestore fetch failed: {e}"); return []

def send_email(sender, password, receiver, df, loc, summary, user):
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"🚧 PotholeAI Report — {loc} — {datetime.now().strftime('%d %b %Y')}"
        msg['From'] = sender; msg['To'] = receiver

        rows = "".join(f"""<tr>
            <td style="padding:8px 12px;border-bottom:1px solid #222;">{r['Filename']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #222;color:{'#ff2d2d' if r['Prediction']=='Pothole' else '#00e676'};font-weight:bold;">{r['Prediction']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #222;">{r['Confidence (%)']}</td>
            <td style="padding:8px 12px;border-bottom:1px solid #222;">{r['Severity']}</td>
        </tr>""" for _,r in df.iterrows())

        html = f"""<html><body style="background:#0d0d0d;color:#e8e0d0;font-family:'Segoe UI',sans-serif;padding:20px;">
        <div style="max-width:600px;margin:0 auto;background:#141414;border-radius:16px;border:1px solid #333;">
            <div style="background:linear-gradient(135deg,#ff6b00,#ff8c00);padding:30px;text-align:center;">
                <h1 style="color:#000;margin:0;font-size:28px;">🛣️ POTHOLE AI</h1>
                <p style="color:#000;margin:8px 0 0;font-size:13px;opacity:0.7;">Road Health Scan Report</p>
            </div>
            <div style="padding:25px;">
                <p style="color:#888;font-size:13px;">👤 Scanned by: <b style="color:#ff8c00;">{user.upper()}</b></p>
                <h2 style="color:#ff8c00;font-size:16px;">📍 {loc}</h2>
                <p style="color:#888;font-size:13px;">📅 {datetime.now().strftime('%d %B %Y at %H:%M')}</p>
                <div style="display:flex;gap:12px;flex-wrap:wrap;margin:15px 0;">
                    {''.join(f'<div style="background:#1a1a1a;border-radius:12px;padding:15px 20px;text-align:center;flex:1;min-width:80px;"><div style="font-size:26px;font-weight:bold;color:{c};">{v}</div><div style="font-size:11px;color:#888;margin-top:4px;">{l}</div></div>' for v,l,c in [(summary['total'],'SCANNED','#ff8c00'),(summary['potholes'],'POTHOLES','#ff2d2d'),(summary['normal'],'NORMAL','#00e676'),(str(summary['avg_conf'])+'%','AVG CONF','#ffaa00')])}
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:13px;">
                    <thead><tr style="background:#1f1f1f;">
                        <th style="padding:10px 12px;text-align:left;color:#888;">File</th>
                        <th style="padding:10px 12px;text-align:left;color:#888;">Result</th>
                        <th style="padding:10px 12px;text-align:left;color:#888;">Conf</th>
                        <th style="padding:10px 12px;text-align:left;color:#888;">Severity</th>
                    </tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                <p style="margin-top:20px;font-size:12px;color:#555;">Generated by PotholeAI · CSV attached</p>
            </div>
        </div></body></html>"""

        msg.attach(MIMEText(html,'html'))
        att = MIMEBase('application','octet-stream')
        att.set_payload(df.to_csv(index=False).encode('utf-8'))
        encoders.encode_base64(att)
        att.add_header('Content-Disposition',f'attachment; filename="report_{datetime.now().strftime("%Y%m%d_%H%M")}.csv"')
        msg.attach(att)

        with smtplib.SMTP_SSL('smtp.gmail.com',465) as s:
            s.login(sender, password)
            s.sendmail(sender, receiver, msg.as_string())
        return True, "Email sent successfully!"
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Auth failed. Check Gmail and App Password."
    except Exception as e:
        return False, f"❌ Error: {e}"


# ─── Location ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📍 Set Scan Location</div>', unsafe_allow_html=True)
use_coords = st.checkbox("📌 Use coordinates instead of location name")
lat = lon = None
location_name = "Unknown"

if use_coords:
    c1,c2 = st.columns(2)
    lat = c1.number_input("Latitude",  value=28.6139, format="%.6f")
    lon = c2.number_input("Longitude", value=77.2090, format="%.6f")
    location_name = f"{lat:.4f}, {lon:.4f}"
    st.markdown(f'<div class="location-box"><div class="location-label">📌 Coordinates set</div><b>{lat}, {lon}</b></div>', unsafe_allow_html=True)
else:
    loc_in = st.text_input("Road name", placeholder="e.g. MG Road Bangalore", label_visibility="collapsed")
    if loc_in:
        with st.spinner("📡 Finding location..."):
            lat, lon = geocode_location(loc_in)
        if lat and lon:
            location_name = loc_in
            st.markdown(f'<div class="location-box"><div class="location-label">✅ Location found</div><b>{loc_in}</b><br><span style="color:#888;font-size:0.8rem;">{lat:.5f}, {lon:.5f}</span></div>', unsafe_allow_html=True)
        else:
            st.error("❌ Location not found. Try adding city name.")

st.caption("✅ Predictions saved to Firestore automatically")


# ─── Uploader ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-heading">📂 Upload Road Images</div>', unsafe_allow_html=True)
st.caption("📱 On mobile: tap below to open camera or gallery")
uploaded_files = st.file_uploader("Upload", type=["jpg","jpeg","png"], accept_multiple_files=True, label_visibility="collapsed")

results = []

if uploaded_files:
    if not lat or not lon:
        st.warning("⚠️ No location set — set a location above to see map pins!")
    st.markdown(f'<div class="section-heading">🔍 Results — {len(uploaded_files)} image(s)</div>', unsafe_allow_html=True)

    num_cols = min(len(uploaded_files), 2)
    rows_list = [uploaded_files[i:i+num_cols] for i in range(0,len(uploaded_files),num_cols)]
    new_pts = []

    for row in rows_list:
        cols = st.columns(len(row))
        for idx, f in enumerate(row):
            with cols[idx]:
                st.image(f, use_column_width=True)
                img = image.load_img(f, target_size=(128,128))
                arr = image.img_to_array(img)
                arr = np.expand_dims(arr,axis=0)/255.0
                raw = float(model.predict(arr,verbose=0)[0][0])
                is_ph = raw > 0.5
                conf  = raw if is_ph else 1-raw
                sev_lbl, sev_css, sev_col = get_severity(conf, is_ph)
                badge = '<span class="badge-danger">⚠ Pothole</span>' if is_ph else '<span class="badge-safe">✓ Normal</span>'
                bar   = "conf-bar-fill-danger" if is_ph else "conf-bar-fill-safe"
                pct   = int(conf*100)

                st.markdown(f"""
<div class="result-card">
    <div class="filename-label">📄 {f.name}</div>
    {badge}
    <div style="margin-top:0.8rem;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#888;">CONFIDENCE</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#ccc;">{pct}%</span>
        </div>
        <div class="conf-bar-bg"><div class="{bar}" style="width:{pct}%;"></div></div>
    </div>
    <div style="margin-top:0.6rem;font-family:'Space Mono',monospace;font-size:0.75rem;color:#888;">
        SEVERITY &nbsp; <span class="{sev_css}">{sev_lbl}</span>
    </div>
</div>""", unsafe_allow_html=True)

                pred_txt = "Pothole" if is_ph else "Normal Road"
                results.append({"Filename":f.name,"Prediction":pred_txt,"Confidence (%)":f"{pct}%","Severity":sev_lbl,
                                 "Location":location_name,"Latitude":lat if lat else "N/A","Longitude":lon if lon else "N/A",
                                 "Scanned By":st.session_state.username,"Timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

                try:
                    save_detection({"filename":f.name,"prediction":pred_txt,"confidence":pct,"severity":sev_lbl,
                                    "location":location_name,"latitude":float(lat) if lat else None,
                                    "longitude":float(lon) if lon else None,
                                    "scanned_by":st.session_state.username,"timestamp":datetime.now().isoformat()})
                except Exception as e:
                    st.warning(f"Firestore save failed: {e}")

                if lat and lon:
                    off = 0.0003
                    new_pts.append({"lat":lat+random.uniform(-off,off),"lon":lon+random.uniform(-off,off),
                                    "filename":f.name,"prediction":pred_txt,"confidence":pct,"severity":sev_lbl,
                                    "severity_color":sev_col,"location":location_name,"timestamp":datetime.now().strftime("%H:%M:%S")})

    if new_pts:       st.session_state.scan_history.extend(new_pts)
    if results:       st.session_state.scan_results = results

    if results:
        total   = len(results)
        ph      = sum(1 for r in results if r["Prediction"]=="Pothole")
        norm    = total-ph
        avg_c   = sum(int(r["Confidence (%)"].replace("%","")) for r in results)/total
        st.markdown(f"""
<div class="summary-box">
    <div class="summary-stat"><div class="number">{total}</div><div class="label">Scanned</div></div>
    <div class="summary-stat"><div class="number" style="color:#ff2d2d">{ph}</div><div class="label">Potholes</div></div>
    <div class="summary-stat"><div class="number" style="color:#00e676">{norm}</div><div class="label">Normal</div></div>
    <div class="summary-stat"><div class="number" style="color:#ffaa00">{avg_c:.0f}%</div><div class="label">Avg Conf</div></div>
</div>""", unsafe_allow_html=True)
        df_csv = pd.DataFrame(results)
        st.download_button("📥 Download Report (CSV)", data=df_csv.to_csv(index=False).encode("utf-8"), file_name="pothole_report.csv", mime="text/csv")


# ─── Email Section ─────────────────────────────────────────────────────────────
if st.session_state.scan_results:
    st.markdown("---")
    st.markdown('<div class="section-heading">📧 Send Report by Email</div>', unsafe_allow_html=True)
    st.markdown('<div class="email-box"><div class="email-title">📤 Send to Authorities</div>', unsafe_allow_html=True)
    receiver = st.text_input("Recipient email", placeholder="authority@example.com")
    if st.button("📧 Send Email Report", use_container_width=True):
        if not sender_email or not sender_password:
            st.error("⚠️ Enter Gmail & App Password in the sidebar!")
        elif not receiver or "@" not in receiver:
            st.error("⚠️ Enter a valid email address!")
        else:
            with st.spinner("📤 Sending..."):
                df_mail = pd.DataFrame(st.session_state.scan_results)
                total   = len(st.session_state.scan_results)
                ph      = sum(1 for r in st.session_state.scan_results if r["Prediction"]=="Pothole")
                avg_c   = sum(int(r["Confidence (%)"].replace("%","")) for r in st.session_state.scan_results)/total
                ok, msg = send_email(sender_email, sender_password, receiver, df_mail, location_name,
                                     {"total":total,"potholes":ph,"normal":total-ph,"avg_conf":int(avg_c)},
                                     st.session_state.username)
            if ok:
                st.markdown(f'<div class="email-success">✅ Report sent to {receiver}!</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="email-error">{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── Session Map ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="map-heading">🗺️ Live Pothole Map (This Session)</div>', unsafe_allow_html=True)
st.caption("Tap a pin to see details.")
cb, cm = st.columns([2,1])
with cb:
    if st.button("🗑️ Clear Pins"): st.session_state.scan_history=[]; st.rerun()
with cm:
    st.metric("🔴 Potholes", sum(1 for p in st.session_state.scan_history if p["prediction"]=="Pothole"))

clat = st.session_state.scan_history[-1]["lat"] if st.session_state.scan_history else (lat or 20.5937)
clon = st.session_state.scan_history[-1]["lon"] if st.session_state.scan_history else (lon or 78.9629)

m = folium.Map(location=[clat,clon], zoom_start=14, tiles="CartoDB dark_matter")
for pt in st.session_state.scan_history:
    em = "⚠️" if pt["prediction"]=="Pothole" else "✅"
    folium.Marker(location=[pt["lat"],pt["lon"]],
        popup=folium.Popup(f'<div style="font-family:monospace;font-size:13px;"><b>{em} {pt["prediction"]}</b><br>📄 {pt["filename"]}<br>📊 {pt["confidence"]}%<br>🔆 {pt["severity"]}<br>📍 {pt["location"]}<br>🕐 {pt["timestamp"]}</div>', max_width=220),
        tooltip=f'{em} {pt["prediction"]} ({pt["confidence"]}%)',
        icon=folium.Icon(color=marker_icon(pt["severity_color"]),icon="exclamation-sign" if pt["prediction"]=="Pothole" else "ok-sign",prefix="glyphicon")
    ).add_to(m)
if not st.session_state.scan_history and lat and lon:
    folium.Marker(location=[lat,lon], tooltip="📍 Selected location", icon=folium.Icon(color="blue",icon="map-marker")).add_to(m)
st_folium(m, width="100%", height=400)


# ─── Firestore Feed ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">🌐 Live Firestore Feed (All Users)</div>', unsafe_allow_html=True)
detections = safe_fetch_detections(flt_limit)

if detections:
    df_all = pd.DataFrame(detections)
    for col in ["severity","prediction","location","timestamp","confidence","latitude","longitude","filename","scanned_by"]:
        if col not in df_all.columns: df_all[col] = None

    df_f = df_all[df_all["severity"].isin(flt_severity) & df_all["prediction"].isin(flt_pred)].copy()
    df_f["timestamp"] = pd.to_datetime(df_f["timestamp"], errors="coerce")
    df_f = df_f.dropna(subset=["timestamp"]).sort_values("timestamp", ascending=False)
    st.caption(f"Showing {len(df_f)} / {len(df_all)} detections")
    st.dataframe(df_f[["timestamp","filename","prediction","confidence","severity","location","scanned_by"]], use_container_width=True, hide_index=True)
    st.download_button("📥 Download Firestore CSV", data=df_f.to_csv(index=False).encode("utf-8"), file_name="firestore_data.csv", mime="text/csv")

    # Analytics
    st.markdown('<div class="section-heading">📊 Live Analytics</div>', unsafe_allow_html=True)
    ca, cb2 = st.columns(2)
    with ca:
        if df_f["severity"].notna().any():
            fig1 = px.histogram(df_f, x="severity", title="Severity Distribution", color="severity",
                color_discrete_map={"High":"#ff2d2d","Medium":"#ffaa00","Low":"#f5c518","Safe":"#00e676"})
            fig1.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", font_color="#e8e0d0")
            st.plotly_chart(fig1, use_container_width=True)
    with cb2:
        if df_f["scanned_by"].notna().any():
            uc = df_f["scanned_by"].value_counts().reset_index()
            uc.columns = ["User","Scans"]
            fig3 = px.bar(uc, x="User", y="Scans", title="Scans Per User", color="Scans", color_continuous_scale="Oranges")
            fig3.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", font_color="#e8e0d0")
            st.plotly_chart(fig3, use_container_width=True)

    df_f["date"] = df_f["timestamp"].dt.date
    trend = df_f.groupby("date").size().reset_index(name="detections")
    if not trend.empty:
        fig2 = px.line(trend, x="date", y="detections", title="Detections Over Time")
        fig2.update_layout(paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d", font_color="#e8e0d0")
        st.plotly_chart(fig2, use_container_width=True)

    # Firestore map
    st.markdown('<div class="section-heading">🗺️ Firestore Map (All Users)</div>', unsafe_allow_html=True)
    pins = df_f.dropna(subset=["latitude","longitude"]).copy()
    if not pins.empty:
        fm = folium.Map(location=[float(pins.iloc[0]["latitude"]),float(pins.iloc[0]["longitude"])], zoom_start=13, tiles="CartoDB dark_matter")
        for _, row in pins.head(200).iterrows():
            em = "⚠️" if str(row["prediction"]).lower().startswith("pothole") else "✅"
            folium.Marker(
                location=[float(row["latitude"]),float(row["longitude"])],
                popup=folium.Popup(f'<div style="font-family:monospace;font-size:13px;"><b>{em} {row.get("prediction")}</b><br>👤 {row.get("scanned_by","unknown")}<br>🔆 {row.get("severity")}<br>📊 {row.get("confidence")}%<br>📍 {row.get("location")}</div>', max_width=250),
                tooltip=f'{em} {row.get("prediction")} ({row.get("confidence")}%)'
            ).add_to(fm)
        st_folium(fm, width="100%", height=450)
    else:
        st.info("No GPS detections in Firestore yet.")
else:
    st.info("No detections in Firestore yet. Upload images to get started.")
