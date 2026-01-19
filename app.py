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
import scipy.ndimage

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="DJ's Ear Pro Music Elite v2", page_icon="🎼", layout="wide")

TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

# Profils Krumhansl-Kessler pour corrélation
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .report-card { 
        padding: 40px; border-radius: 25px; text-align: center; color: white; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.2); color: #f87171;
        padding: 15px; border-radius: 12px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-size: 1.1em;
    }
    .metric-box {
        background: #1a1c24; border-radius: 15px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT ---

def apply_enhanced_filters(y, sr):
    """Isolation harmonique avancée et filtrage spectral"""
    # HPSS agressif pour éliminer la batterie qui pollue les notes
    y_harm, _ = librosa.effects.hpss(y, margin=(8.0, 1.0))
    y_harm = librosa.util.normalize(y_harm)
    
    # Filtrage passe-bande focalisé sur les fondamentales instrumentales
    nyq = 0.5 * sr
    low, high = 150 / nyq, 3500 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def solve_key_precision(chroma_vector):
    """Analyse par corrélation de Pearson avec renforcement des triades"""
    best_score = -1
    res = {"key": "Inconnu", "score": 0}
    
    # Normalisation pour l'invariance d'énergie
    cv = (chroma_vector - np.mean(chroma_vector)) / (np.std(chroma_vector) + 1e-6)
    
    for mode in ["major", "minor"]:
        for i in range(12):
            profile = np.roll(PROFILES[mode], i)
            profile = (profile - np.mean(profile)) / (np.std(profile) + 1e-6)
            
            # Corrélation statistique
            corr_score = np.corrcoef(cv, profile)[0, 1]
            
            # Bonus de structure harmonique (Tierce et Quinte)
            third_idx = (i + 3) % 12 if mode == "minor" else (i + 4) % 12
            fifth_idx = (i + 7) % 12
            triad_presence = (chroma_vector[i] * 1.0) + (chroma_vector[third_idx] * 0.7) + (chroma_vector[fifth_idx] * 0.5)
            
            final_score = corr_score + (0.1 * triad_presence)

            if final_score > best_score:
                best_score = final_score
                res = {"key": f"{NOTES_LIST[i]} {mode}", "score": corr_score}
    return res

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    # 1. Estimation fine du tuning (résolution 36 bins)
    tuning = librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=36)
    y_filt = apply_enhanced_filters(y, sr)
    
    # 2. Extraction Chroma CQT haute résolution
    chroma_complex = librosa.feature.chroma_cqt(
        y=y_filt, sr=sr, tuning=tuning, bins_per_octave=36, n_octaves=6
    )
    
    # 3. Lissage temporel (Median Filter) pour éliminer les artéfacts
    chroma_smooth = scipy.ndimage.median_filter(chroma_complex, size=(1, 9))
    global_chroma_avg = np.mean(chroma_smooth, axis=1)

    duration = librosa.get_duration(y=y, sr=sr)
    step = 5 # Fenêtre plus courte pour plus de précision temporelle
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - step, step):
        # Découpe du chroma lisse au lieu du signal brut pour la cohérence
        idx_start = librosa.time_to_frames(start, sr=sr)
        idx_end = librosa.time_to_frames(start+step, sr=sr)
        c_seg = np.mean(chroma_smooth[:, idx_start:idx_end], axis=1)
        
        if np.max(c_seg) < 0.1: continue
        
        res = solve_key_precision(c_seg)
        
        # Pondération : on donne plus de poids au corps du morceau qu'aux transitions
        weight = 1.2 if (duration*0.2 < start < duration*0.8) else 0.8
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    # Résultats principaux
    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    main_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    # Détection de modulation
    target_key, target_conf, modulation_detected = None, 0, False
    if len(most_common) > 1:
        second_key = most_common[1][0]
        count_second = sum(1 for t in timeline if t['Note'] == second_key)
        if count_second > (len(timeline) * 0.20): 
            modulation_detected = True
            target_key = second_key
            target_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == second_key]) * 100)

    # Tempo haute précision
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    output = {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": main_conf,
        "tempo": int(float(tempo)),
        "tuning_hz": round(440 * (2**(tuning/12)), 1),
        "pitch_offset": round(tuning, 2),
        "timeline": timeline, "chroma": global_chroma_avg,
        "modulation": modulation_detected, 
        "target_key": target_key,
        "target_conf": target_conf,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    del y, y_filt, y_perc; gc.collect()
    return output

# --- FONCTIONS UI (PIANO & TELEGRAM) ---

def get_piano_js(button_id, key_name):
    if not key_name or " " not in key_name: return ""
    n, mode = key_name.split()
    return f"""
    document.getElementById('{button_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const chord = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        chord.forEach((interval) => {{
            const baseFreq = freqs['{n}'] * Math.pow(2, interval/12);
            [1, 2].forEach((h) => {{
                const osc = ctx.createOscillator(); const g = ctx.createGain();
                osc.type = h === 1 ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(baseFreq * h, ctx.currentTime);
                g.gain.setValueAtTime(0, ctx.currentTime);
                g.gain.linearRampToValueAtTime(0.1/h, ctx.currentTime + 0.05);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                osc.connect(g); g.connect(ctx.destination);
                osc.start(); osc.stop(ctx.currentTime + 1.5);
            }});
        }});
    }};
    """

def send_telegram_expert(data, fig_timeline, fig_radar):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    mod_text = f"⚠️ *MODULATION: {data['target_key']}* ({data['target_camelot']})\n" if data['modulation'] else "💎 *STABILITÉ HARMONIQUE*\n"
    msg = (f"🎼 *PRO REPORT:* `{data['name']}`\n"
           f"━━━━━━━━━━━━━━━━━━━━\n"
           f"✅ *TONALE:* `{data['key'].upper()}` ({data['camelot']})\n"
           f"🎯 *CONF:* `{data['conf']}%` | BPM: `{data['tempo']}`\n"
           f"{mod_text}"
           f"━━━━━━━━━━━━━━━━━━━━")
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        for fig in [fig_timeline, fig_radar]:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", data={"chat_id": CHAT_ID}, files={"photo": img_bytes})
    except: pass

# --- INTERFACE PRINCIPALE ---
st.title("🎧 DJ's Ear Pro Elite v2")
files = st.file_uploader("📂 Audio (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        data = analyze_full_engine(f.read(), f.name)
        with st.expander(f"📊 {data['name']}", expanded=True):
            bg = "linear-gradient(135deg, #0f172a, #1e3a8a)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #7f1d1d)"
            st.markdown(f"""<div class="report-card" style="background:{bg};">
                <h1 style="font-size:5em; margin:0;">{data['key'].upper()}</h1>
                <p style="font-size:1.5em;">{data['camelot']} | CONF: {data['conf']}% | {data['tempo']} BPM</p>
                {f"<div class='modulation-alert'>MODULATION VERS {data['target_key'].upper()}</div>" if data['modulation'] else ""}
                </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1: st.markdown(f"<div class='metric-box'><b>TUNING</b><br>{data['tuning_hz']} Hz</div>", unsafe_allow_html=True)
            with c2: 
                btn_id = f"p_{hash(f.name)}"
                components.html(f"""<button id="{btn_id}" style="width:100%; height:80px; background:#4F46E5; color:white; border:none; border-radius:10px; cursor:pointer;">🎹 TEST ACCORD</button><script>{get_piano_js(btn_id, data['key'])}</script>""", height=100)
            with c3: st.markdown(f"<div class='metric-box'><b>OFFSET</b><br>{data['pitch_offset']}c</div>", unsafe_allow_html=True)

            gl, gr = st.columns([2, 1])
            with gl:
                fig_l = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                st.plotly_chart(fig_l, use_container_width=True)
            with gr:
                fig_r = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself'))
                fig_r.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)))
                st.plotly_chart(fig_r, use_container_width=True)
            
            send_telegram_expert(data, fig_l, fig_r)

if st.sidebar.button("🧹 Clear"):
    st.cache_data.clear(); st.rerun()
