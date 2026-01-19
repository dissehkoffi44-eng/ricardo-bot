import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from datetime import datetime

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3 ELITE", page_icon="🎯", layout="wide")

# Récupération des secrets Telegram
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for n in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 25px;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.2); color: #f87171;
        padding: 15px; border-radius: 12px; border: 1px solid #ef4444;
        margin-top: 15px; font-weight: bold; font-family: 'JetBrains Mono', monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
        height: 100%;
    }
    .status-badge {
        background: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL ---

def apply_pro_filters(y, sr):
    """Filtre harmonique + Passe-bande chirurgical (Code 2)"""
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 4500/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_triad_indices(key_name):
    """Récupère les indices (0-11) des notes de la triade (Fondamentale, Tierce, Quinte)"""
    note_root, mode = key_name.split()
    root_idx = NOTES_LIST.index(note_root)
    third_interval = 4 if mode == "major" else 3
    return [root_idx, (root_idx + third_interval) % 12, (root_idx + 7) % 12]

def triad_arbitration(key1, key2, global_chroma):
    """Arbitrage par comparaison entre les triades et les 4 notes dominantes réelles"""
    # On identifie les 4 notes les plus fortes du spectre complet
    top_4_real = np.argsort(global_chroma)[-4:]
    
    def score_key(k):
        triad = get_triad_indices(k)
        # On compte combien de notes de la triade sont dans le TOP 4 réel
        return sum(1 for note in triad if note in top_4_real)

    s1 = score_key(key1)
    s2 = score_key(key2)

    # Si la modulation (key2) a un meilleur score de triade, on switch
    if s2 > s1:
        return key2, key1, True 
    return key1, key2, False

def solve_key_sniper(chroma_vec):
    """Scoring basé sur les profils Krumhansl (Code 1 amélioré)"""
    best_score, best_k = -1, ""
    for mode in ["major", "minor"]:
        for i in range(12):
            score = np.corrcoef(chroma_vec, np.roll(PROFILES["krumhansl"][mode], i))[0, 1]
            if score > best_score:
                best_score = score
                best_k = f"{NOTES_LIST[i]} {mode}"
    return best_k, best_score

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_pro_filters(y, sr)
    
    # Analyse segmentée
    step, timeline, votes = 6, [], Counter()
    for start in range(0, int(duration) - step, 3):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.01: continue
        
        c_seg = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12)
        c_avg = np.mean(c_seg, axis=1)
        
        key, score = solve_key_sniper(c_avg)
        # Poids renforcé sur le début et la fin (Intro/Outro)
        weight = 1.5 if (start < 15 or start > (duration - 15)) else 1.0
        votes[key] += int(score * 100 * weight)
        timeline.append({"Temps": start, "Note": key, "Conf": score})

    # Synthèse et Arbitrage
    global_chroma = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    target_key = None
    triad_fix = False

    if len(most_common) > 1:
        potential_mod = most_common[1][0]
        # Si la modulation représente plus de 20% de l'énergie totale
        if (votes[potential_mod] / sum(votes.values())) > 0.20:
            main_key, target_key, triad_fix = triad_arbitration(main_key, potential_mod, global_chroma)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res = {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100),
        "tempo": int(float(tempo)), "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "chroma": global_chroma,
        "modulation": True if target_key else False,
        "target_key": target_key, "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "triad_validated": triad_fix, "name": file_name
    }
    del y, y_filt; gc.collect()
    return res

def get_chord_js(btn_id, key_str):
    if not key_str or " " not in key_str: return ""
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const chord = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        chord.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.12, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.8);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 1.8);
        }});
    }};
    """

def send_telegram(data):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    mod_info = f"\n⚠️ MODULATION: {data['target_key']} ({data['target_camelot']})" if data['modulation'] else ""
    fix_info = "\n✅ VALIDATION TRIADIQUE APPLIQUÉE" if data['triad_validated'] else ""
    msg = (f"🎯 *SNIPER M3 ELITE REPORT*\n"
           f"📂 `{data['name']}`\n"
           f"🎹 TONALITÉ: *{data['key'].upper()}*\n"
           f"🌀 CAMELOT: `{data['camelot']}`\n"
           f"🔥 CONF: `{data['conf']}%`\n"
           f"{mod_info}{fix_info}\n"
           f"⏱ BPM: `{data['tempo']}` | `{data['tuning']}Hz`")
    try: requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- INTERFACE UTILISATEUR ---
st.title("🎯 RCDJ228 SNIPER M3 ELITE")
st.markdown("### Système Militaire d'Analyse Harmonique par Arbitrage de Triades")

files = st.file_uploader("📥 Déposer des fichiers audio", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse Sniper de {f.name}..."):
            data = analyze_full_engine(f.read(), f.name)
        
        # Dashboard Résultat
        color = "linear-gradient(135deg, #064e3b, #065f46)" if data['conf'] > 85 else "linear-gradient(135deg, #1e1b4b, #111827)"
        st.markdown(f"""
            <div class="report-card" style="background:{color};">
                <p style="opacity:0.6; letter-spacing:3px;">SNIPER ENGINE ELITE v5.0 <span class="status-badge">ACTIVE</span></p>
                <h1 style="font-size:6em; margin:15px 0; font-weight:900;">{data['key'].upper()}</h1>
                <p style="font-size:1.8em;">CAMELOT: <b>{data['camelot']}</b> &nbsp; | &nbsp; CONFIANCE: <b>{data['conf']}%</b></p>
                {"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : " + data['target_key'].upper() + " (" + data['target_camelot'] + ")</div>" if data['modulation'] else ""}
                {"<p style='color:#10b981; margin-top:15px; font-weight:bold;'>✓ Tonalité arbitrée et validée par analyse spectrale des triades</p>" if data['triad_validated'] else ""}
            </div>
        """, unsafe_allow_html=True)

        # Métriques secondaires
        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#3b82f6;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
        with m3:
            btn_id = f"play_{hash(f.name)}"
            components.html(f"""
                <button id="{btn_id}" style="width:100%; height:90px; background:linear-gradient(45deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1em;">🎹 JOUER L'ACCORD</button>
                <script>{get_chord_js(btn_id, data['key'])}</script>
            """, height=100)

        # Visualisations
        col_l, col_r = st.columns([2, 1])
        with col_l:
            fig_tl = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER}, title="Flux Harmonique (Timeline)")
            fig_tl.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_tl, use_container_width=True)
        with col_r:
            fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_radar.update_layout(template="plotly_dark", height=350, polar=dict(radialaxis=dict(visible=False)), title="Empreinte Chromatique", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_radar, use_container_width=True)

        send_telegram(data)
        st.write("---")

with st.sidebar:
    st.header("Sniper Control")
    if st.button("🧹 Vider la mémoire"):
        st.cache_data.clear()
        st.rerun()
