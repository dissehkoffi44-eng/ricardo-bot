# RCDJ228 SNIPER M3 - VERSION ULTIME (MULTI-FILES & PRECISION)
import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import os
import requests
import gc
import json
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
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

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
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "bellman": {
        "major": [16.8, 0.86, 12.95, 1.41, 13.49, 11.93, 1.25, 16.74, 1.56, 12.81, 1.89, 12.44],
        "minor": [18.16, 0.69, 12.99, 13.34, 1.07, 11.15, 1.38, 17.2, 13.62, 1.27, 12.79, 2.4]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0b0e14; }
    .report-card { 
        padding: 40px; border-radius: 30px; text-align: center; color: white; 
        border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 15px 45px rgba(0,0,0,0.6);
        margin-bottom: 20px;
    }
    .file-header {
        background: #1f2937; color: #10b981; padding: 12px 20px; border-radius: 12px;
        font-family: 'JetBrains Mono', monospace; font-weight: bold; margin-bottom: 10px;
        border-left: 6px solid #10b981; display: flex; justify-content: space-between;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 20px; text-align: center; border: 1px solid #30363d;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.15); color: #f87171; padding: 15px; border-radius: 15px; border: 1px solid #ef4444; margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE DE CALCUL ---
def apply_sniper_filters(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=8.0)
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 4500/nyq], btype='band')
    return lfilter(b, a, y_harm)

def solve_key_sniper(chroma_vector):
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    best_overall_score, best_key = -1, "Unknown"
    profile_results = []

    for p_name, p_data in PROFILES.items():
        p_best_score, p_best_key = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                # Bonus Tierce/Quinte
                third = (i + 4) % 12 if mode == "major" else (i + 3) % 12
                score += (cv[third] * 0.25) + (cv[(i + 7) % 12] * 0.15)
                # Pénalité hors-gamme
                if mode == "major":
                    for n in [(i+1)%12, (i+3)%12, (i+8)%12, (i+10)%12]:
                        if cv[n] > 0.4: score -= 0.15
                
                if score > p_best_score:
                    p_best_score, p_best_key = score, f"{NOTES_LIST[i]} {mode}"
        
        profile_results.append(p_best_key)
        if p_best_score > best_overall_score:
            best_overall_score, best_key = p_best_score, p_best_key

    if Counter(profile_results).most_common(1)[0][1] == 3:
        best_overall_score *= 1.2
    
    return {"key": best_key, "score": best_overall_score}

def process_file(file_bytes, file_name, progress_bar, status_text):
    try:
        status_text.text(f"Chargement de {file_name}...")
        ext = file_name.split('.')[-1].lower()
        if ext == 'm4a':
            audio = AudioSegment.from_file(io.BytesIO(file_bytes), format="m4a")
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            if audio.channels == 2: samples = samples.reshape((-1, 2)).mean(axis=1)
            y, sr = samples / (2**15), audio.frame_rate
        else:
            with io.BytesIO(file_bytes) as buf:
                y, sr = librosa.load(buf, sr=22050, mono=True)
        
        duration = librosa.get_duration(y=y, sr=sr)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        y_filt = apply_sniper_filters(y, sr)
        
        step, timeline, votes = 2, [], Counter()
        segments = list(range(0, max(1, int(duration) - step), 1))
        
        for idx, start in enumerate(segments):
            prog = int((idx / len(segments)) * 100)
            progress_bar.progress(prog)
            status_text.text(f"Analyse Harmonique : {start}s / {int(duration)}s")
            
            idx_s, idx_e = int(start * sr), int((start + step) * sr)
            seg = y_filt[idx_s:idx_e]
            if len(seg) < 1000 or np.max(np.abs(seg)) < 0.01: continue
            
            c_raw = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=36)
            res = solve_key_sniper(np.median(c_raw, axis=1))
            
            if res['score'] < 0.75: continue
            weight = 2.5 if (start < 15 or start > (duration - 20)) else 1.0
            votes[res['key']] += int(res['score'] * 100 * weight)
            timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

        if not votes: return None

        most_common = votes.most_common(2)
        final_key = most_common[0][0]
        final_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == final_key]) * 100)
        mod_detected = len(most_common) > 1 and (votes[most_common[1][0]] / sum(votes.values())) > 0.30
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        res_obj = {
            "key": final_key, "camelot": CAMELOT_MAP.get(final_key, "??"),
            "conf": min(final_conf + 10, 99), "tempo": int(float(tempo)),
            "tuning": round(440 * (2**(tuning/12)), 1), "timeline": timeline,
            "modulation": mod_detected, "target_key": most_common[1][0] if mod_detected else None,
            "chroma": np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, bins_per_octave=36), axis=1).tolist(),
            "name": file_name
        }
        
        del y, y_filt; gc.collect()
        return res_obj
    except Exception as e:
        st.error(f"Erreur sur {file_name} : {e}")
        return None

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
            g.gain.linearRampToValueAtTime(0.15, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 2.0);
        }});
    }}; """

# --- UI PRINCIPALE ---
st.title("🎯 RCDJ228 SNIPER M3 - VERSION ULTIME")

files = st.file_uploader("📂 Déposez vos fichiers (Traitement Multiples)", type=['mp3','wav','flac','m4a'], accept_multiple_files=True)

if files:
    total = len(files)
    global_prog = st.progress(0)
    global_status = st.empty()
    results_area = st.container()

    for idx, f in enumerate(files):
        # Indexation et Progression Globale
        current_idx = idx + 1
        global_prog.progress(current_idx / total)
        global_status.markdown(f"**Analyse en cours : {current_idx} / {total}** | `{f.name}`")

        with st.expander(f"🔍 Scan de : {f.name}", expanded=True):
            p_bar = st.progress(0)
            p_text = st.empty()
            data = process_file(f.getvalue(), f.name, p_bar, p_text)
            
            if data:
                p_text.success(f"Analyse de {f.name} terminée avec succès.")
                with results_area:
                    st.markdown(f"""<div class='file-header'>
                        <span>📄 [{current_idx}/{total}] {data['name']}</span>
                        <span style='color:#58a6ff;'>BPM: {data['tempo']} | Hz: {data['tuning']}</span>
                    </div>""", unsafe_allow_html=True)
                    
                    color = "linear-gradient(135deg, #065f46, #064e3b)" if data['conf'] > 88 else "linear-gradient(135deg, #1e293b, #0f172a)"
                    st.markdown(f"""
                        <div class="report-card" style="background:{color};">
                            <h1 style="font-size:5.5em; margin:10px 0; font-weight:900;">{data['key'].upper()}</h1>
                            <p style="font-size:1.5em; opacity:0.9;">CAMELOT: <b>{data['camelot']}</b> | CONFIANCE: <b>{data['conf']}%</b></p>
                            {f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : {data['target_key'].upper()}</div>" if data['modulation'] else ""}
                        </div> """, unsafe_allow_html=True)
                    
                    m1, m2, m3 = st.columns(3)
                    with m1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:2em; color:#10b981;'>{data['tempo']}</span><br>BPM</div>", unsafe_allow_html=True)
                    with m2: st.markdown(f"<div class='metric-box'><b>ACCORDAGE</b><br><span style='font-size:2em; color:#58a6ff;'>{data['tuning']}</span><br>Hz</div>", unsafe_allow_html=True)
                    with m3:
                        bid = f"pl_{idx}"
                        components.html(f"""<button id="{bid}" style="width:100%; height:95px; background:linear-gradient(45deg, #10b981, #059669); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold;">🔊 TESTER</button>
                                        <script>{get_chord_js(bid, data['key'])}</script>""", height=110)

                    c1, c2 = st.columns([2, 1])
                    with c1:
                        fig_tl = px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER})
                        fig_tl.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig_tl, use_container_width=True, key=f"tl_{idx}")
                    with c2:
                        fig_rd = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
                        fig_rd.update_layout(template="plotly_dark", height=250, polar=dict(radialaxis=dict(visible=False)), margin=dict(l=20, r=20, t=10, b=10))
                        st.plotly_chart(fig_rd, use_container_width=True, key=f"rd_{idx}")
                    st.markdown("<br>", unsafe_allow_html=True)

    global_status.success(f"✅ Mission terminée ! {total} fichiers analysés.")

with st.sidebar:
    st.header("Sniper Control")
    st.info("Cette version gère l'indexation multiple et la mémoire optimisée.")
    if st.button("🗑️ Vider le cache"):
        st.cache_data.clear()
        st.rerun()
