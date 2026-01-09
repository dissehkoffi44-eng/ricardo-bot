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

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="Absolute Key Detector V4.2 - Precision Mode", page_icon="🎼", layout="wide")

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

# Profils enrichis pour une meilleure distinction Major/Minor
PROFILES = {
    "shaath": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .report-card { 
        background: linear-gradient(135deg, #0f172a, #1e1b4b); 
        padding: 40px; border-radius: 25px; text-align: center; color: white; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.1); color: #f87171;
        padding: 15px; border-radius: 12px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-size: 1.1em;
    }
    .metric-box {
        background: #1a1c24; border-radius: 15px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT ---

def apply_filters(y, sr):
    # Séparation Harmonique plus agressive pour isoler les notes des accords (margin plus élevé)
    y_harm, _ = librosa.effects.hpss(y, margin=(4.0, 1.0))
    y_harm = librosa.effects.preemphasis(y_harm)
    
    # Filtre passe-bande resserré sur la zone mélodique fondamentale (100Hz - 3000Hz)
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 3000/nyq], btype='band')
    y_clean = lfilter(b, a, y_harm)
    return y_clean

def solve_key(chroma_vector):
    best_score = -1
    res = {"key": "Inconnu", "score": 0}
    
    # Normalisation pour l'analyse de corrélation
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                # Corrélation standard
                rotated_profile = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, rotated_profile)[0, 1]
                
                # --- LOGIQUE DE DÉCISION MAJEUR/MINEUR ---
                # On identifie la tierce (index 3 pour mineur, 4 pour majeur)
                third_idx = (i + 3) % 12 if mode == "minor" else (i + 4) % 12
                # On identifie la quinte (index 7)
                fifth_idx = (i + 7) % 12
                
                # Bonus si la tierce caractéristique est réellement présente dans l'audio
                # Cela évite que "A Major" soit confondu avec "A Minor" par erreur
                tierce_presence = cv[third_idx]
                quinte_presence = cv[fifth_idx]
                
                # On ajuste le score final : corrélation + poids des notes piliers
                final_score = score + (0.15 * tierce_presence) + (0.05 * quinte_presence)

                if final_score > best_score:
                    best_score = final_score
                    res = {"key": f"{NOTES_LIST[i]} {mode}", "score": score}
    return res

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # Prétraitement
    y_filt = apply_filters(y, sr)
    
    step = 6 
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - step, step):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.01: continue
        
        # Extraction Chroma CQT haute résolution (36 bins/octave)
        chroma = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, bins_per_octave=36, threshold=0.1)
        chroma_avg = np.mean(chroma, axis=1)
        
        result = solve_key(chroma_avg)
        votes[result['key']] += int(result['score'] * 100)
        timeline.append({"Temps": start, "Note": result['key'], "Conf": result['score']})

    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    
    target_key = None
    modulation_detected = False
    
    if len(most_common) > 1:
        second_key = most_common[1][0]
        unique_keys_in_flow = [t['Note'] for t in timeline]
        if unique_keys_in_flow.count(second_key) > (len(timeline) * 0.18):
            modulation_detected = True
            target_key = second_key

    avg_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    # Tempo basé sur la version percursive isolée
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    full_chroma = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)
    
    output = {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": avg_conf, "tempo": int(float(tempo)),
        "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "chroma": full_chroma,
        "modulation": modulation_detected, 
        "target_key": target_key,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    
    del y, y_filt, y_perc
    gc.collect()
    return output

def get_piano_js(button_id, key_name):
    if not key_name or " " not in key_name: return ""
    n, mode = key_name.split()
    return f"""
    document.getElementById('{button_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const isMinor = '{mode}' === 'minor';
        const chord = isMinor ? [0, 3, 7, 12] : [0, 4, 7, 12];
        chord.forEach((interval) => {{
            const baseFreq = freqs['{n}'] * Math.pow(2, interval/12);
            [1, 2, 3].forEach((harmonic, index) => {{
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = index === 0 ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(baseFreq * harmonic, ctx.currentTime);
                gain.gain.setValueAtTime(0, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0.1 / harmonic, ctx.currentTime + 0.05);
                gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 2.0);
                osc.connect(gain); gain.connect(ctx.destination);
                osc.start(); osc.stop(ctx.currentTime + 2.0);
            }});
        }});
    }};
    """

st.title("🎧 ABSOLUTE KEY DETECTOR V4.2")
st.subheader("Analyse de précision : Distinction Majeur/Mineur renforcée")

uploaded_files = st.file_uploader("📂 Glissez vos fichiers audio ici", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        with st.spinner(f"Analyse harmonique de {f.name}..."):
            data = analyze_full_engine(f.read(), f.name)
        
        with st.expander(f"📊 RÉSULTATS : {data['name']}", expanded=True):
            bg_color = "linear-gradient(135deg, #0f172a, #1e3a8a)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #7f1d1d)"
            modulation_html = f"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE <br><span style='font-size:1.3em;'>Vers : {data['target_key'].upper()} ({data['target_camelot']})</span></div>" if data['modulation'] else ""

            st.markdown(f"""
                <div class="report-card" style="background:{bg_color};">
                    <p style="text-transform:uppercase; letter-spacing:2px; opacity:0.7;">Tonalité Dominante</p>
                    <h1 style="font-size:6em; margin:10px 0;">{data['key'].upper()}</h1>
                    <p style="font-size:1.8em;">CAMELOT : <b>{data['camelot']}</b> | CONFIANCE : <b>{data['conf']}%</b></p>
                    {modulation_html}
                </div>
            """, unsafe_allow_html=True)

            st.write("---")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:1.8em;'>{data['tempo']} BPM</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box' style='margin-top:10px;'><b>ACCORDAGE</b><br><span style='font-size:1.2em;'>{data['tuning']} Hz</span></div>", unsafe_allow_html=True)
            with m2:
                st.markdown(f"<b>ACCORD {data['key'].upper()}</b>", unsafe_allow_html=True)
                uid_main = f"btn_main_{f.name.replace('.','').replace(' ','')}"
                components.html(f"""<button id="{uid_main}" style="width:100%; height:100px; background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🎹 JOUER L'ACCORD</button><script>{get_piano_js(uid_main, data['key'])}</script>""", height=120)
            with m3:
                if data['modulation']:
                    st.markdown(f"<b>ACCORD {data['target_key'].upper()}</b>", unsafe_allow_html=True)
                    uid_mod = f"btn_mod_{f.name.replace('.','').replace(' ','')}"
                    components.html(f"""<button id="{uid_mod}" style="width:100%; height:100px; background:linear-gradient(90deg, #ef4444, #b91c1c); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🎹 JOUER L'ACCORD</button><script>{get_piano_js(uid_mod, data['target_key'])}</script>""", height=120)
                else:
                    st.markdown("<div style='height:120px; display:flex; align-items:center; justify-content:center; opacity:0.3; border:2px dashed #444; border-radius:15px;'>Pas de Modulation</div>", unsafe_allow_html=True)

            c_left, c_right = st.columns([2, 1])
            with c_left:
                st.markdown("#### 📈 Flux Harmonique")
                df_tl = pd.DataFrame(data['timeline'])
                fig_line = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER}, color_discrete_sequence=['#818cf8'])
                fig_line.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=350)
                st.plotly_chart(fig_line, use_container_width=True)
            with c_right:
                st.markdown("#### 🌀 Profil de Fréquences")
                fig_radar = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#818cf8'))
                fig_radar.update_layout(template="plotly_dark", polar=dict(radialaxis=dict(visible=False)), margin=dict(l=30,r=30,t=30,b=30), height=350)
                st.plotly_chart(fig_radar, use_container_width=True)

            if TELEGRAM_TOKEN and CHAT_ID:
                try:
                    mod_txt = f"⚠️ Modulation vers {data['target_key']}" if data['modulation'] else "✅ Stable"
                    msg = (f"🎹 *RAPPORT PRECISION V4.2*\n📂 `{data['name']}`\n\n"
                           f"*Tonalité:* `{data['key']}`\n*Camelot:* `{data['camelot']}`\n"
                           f"*Stabilité:* {mod_txt}\n*Confiance:* `{data['conf']}%`\n"
                           f"*Tempo:* `{data['tempo']} BPM`")
                    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
                except: pass

    if st.sidebar.button("🧹 Vider le cache mémoire"):
        st.cache_data.clear()
        st.rerun()
else:
    st.info("Glissez vos fichiers audio pour commencer l'analyse.")
