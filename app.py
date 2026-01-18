# RCDJ228 SNIPER M3 - VERSION "ROBUSTE" (FIX QUALITÉ AUDIO)
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from collections import Counter
import io
import os
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter
from pydub import AudioSegment

# --- FORCE FFMEG PATH (WINDOWS FIX) ---
if os.path.exists(r'C:\ffmpeg\bin'):
    os.environ["PATH"] += os.pathsep + r'C:\ffmpeg\bin'

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="RCDJ228 SNIPER M3", page_icon="🎯", layout="wide")

# Récupération des secrets
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "sniper_triads": { 
        "major": [1.0, 0, 0, 0, 1.0, 0, 0, 1.0, 0, 0, 0, 0], 
        "minor": [1.0, 0, 0, 1.0, 0, 0, 0, 1.0, 0, 0, 0, 0]
    },
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { padding: 40px; border-radius: 30px; text-align: center; color: white; border: 1px solid rgba(99, 102, 241, 0.3); margin-bottom: 20px; }
    .file-header { background: #1f2937; color: #10b981; padding: 10px 20px; border-radius: 10px; font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px; border-left: 5px solid #10b981; }
    .metric-box { background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d; height: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL AMÉLIORÉS ---
def apply_sniper_filters(y, sr):
    # Séparation Harmonique (Isole les instruments mélodiques du bruit/percussions)
    y_harm = librosa.effects.harmonic(y, margin=3.0)
    # Filtre passe-bande plus serré (60Hz - 3500Hz) pour éliminer le bruit de compression
    nyq = 0.5 * sr
    low, high = 60/nyq, 3500/nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def solve_key_sniper(cv, bv):
    best_overall_score = -1
    best_key = "Unknown"
    
    cv = (cv - cv.min()) / (cv.max() - cv.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                reference = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, reference)[0, 1]
                
                # Bonus "Sniper" pour les intervalles de Quinte et Tierce
                if mode == "major":
                    if cv[(i + 7) % 12] > 0.6 and cv[(i + 4) % 12] > 0.4: score *= 1.2
                else:
                    if cv[(i + 7) % 12] > 0.6 and cv[(i + 3) % 12] > 0.4: score *= 1.2
                
                if score > best_overall_score:
                    best_overall_score = score
                    best_key = f"{NOTES_LIST[i]} {mode}"
    return {"key": best_key, "score": best_overall_score}

def process_audio_precision(file_bytes, file_name, _progress_callback=None):
    try:
        # Chargement robuste
        with io.BytesIO(file_bytes) as buf:
            y, sr = librosa.load(buf, sr=22050, mono=True)
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_sniper_filters(y, sr)

    step, timeline = 1, [] # Step réduit à 1s pour plus de données de lissage
    segments = list(range(0, max(1, int(duration) - 2), step))
    
    for idx, start in enumerate(segments):
        if _progress_callback:
            _progress_callback(int((idx / len(segments)) * 100), f"Scan...")

        idx_start, idx_end = int(start * sr), int((start + 2) * sr)
        seg = y_filt[idx_start:idx_end]
        if len(seg) < sr or np.max(np.abs(seg)) < 0.02: continue
        
        # RÉSOLUTION ACCRUE : 36 bins par octave pour capturer les nuances des fichiers compressés
        c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=36)
        c_avg = np.mean(c_raw, axis=1)
        
        res = solve_key_sniper(c_avg, None)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    if not timeline: return None

    # --- LISSAGE PAR CONSENSUS (MODE GLISSANT) ---
    df_time = pd.DataFrame(timeline)
    # On prend la note la plus fréquente sur une fenêtre de 5 secondes pour éliminer les erreurs de qualité
    df_time['Note'] = df_time['Note'].rolling(window=5, center=True).apply(lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else x[0], raw=False).fillna(df_time['Note'])
    
    final_votes = Counter(df_time['Note'])
    final_key = final_votes.most_common(1)[0][0]
    final_conf = int(df_time[df_time['Note'] == final_key]['Conf'].mean() * 100)
    
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    res_obj = {
        "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
        "conf": min(final_conf, 99), "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1), "name": file_name
    }

    # Telegram Notification (Silencieuse)
    if TELEGRAM_TOKEN and CHAT_ID:
        try:
            msg = f"🎯 *SNIPER M3*\n`{file_name}`\n🎹 *{final_key.upper()}* ({res_obj['camelot']})\n✅ Conf: {res_obj['conf']}%"
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
        except: pass

    return res_obj

def get_chord_js(btn_id, key_str):
    note, mode = key_str.split()
    return f"""
    document.getElementById('{btn_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const intervals = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(i => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{note}'] * Math.pow(2, i/12), ctx.currentTime);
            g.gain.setValueAtTime(0, ctx.currentTime);
            g.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 1.5);
        }});
    }}; """

# --- INTERFACE ---
st.title("🎯 RCDJ228 SNIPER M3")
files = st.file_uploader("📂 Audio (MP3, WAV, M4A)", type=['mp3','wav','m4a'], accept_multiple_files=True)

if files:
    for i, f in enumerate(reversed(files)):
        with st.status(f"Analyse : {f.name}...") as status:
            data = process_audio_precision(f.getvalue(), f.name, _progress_callback=lambda v, m: None)
            status.update(label=f"Analyse terminée", state="complete")

        if data:
            st.markdown(f"<div class='file-header'>{data['name']}</div>", unsafe_allow_html=True)
            color = "#065f46" if data['conf'] > 80 else "#1e293b"
            st.markdown(f"""<div class="report-card" style="background:{color};">
                <h1 style="font-size:5em; margin:0;">{data['key'].upper()}</h1>
                <p>CAMELOT: {data['camelot']} | CONFIANCE: {data['conf']}%</p>
            </div>""", unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-box'><b>TEMPO</b><br>{data['tempo']} BPM</div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br>{data['tuning']} Hz</div>", unsafe_allow_html=True)
            with c3:
                bid = f"btn_{i}"
                components.html(f"""<button id="{bid}" style="width:100%; height:70px; background:#4F46E5; color:white; border:none; border-radius:10px; cursor:pointer;">🔊 TESTER</button><script>{get_chord_js(bid, data['key'])}</script>""", height=80)
