import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests
import gc
from scipy.signal import butter, lfilter

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M3 PRO", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

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

# --- STYLES CSS PERSONNALISÉS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { 
        background: #1a1c24; padding: 15px; border-radius: 15px; border: 1px solid #333; 
        text-align: center; min-height: 110px; display: flex; flex-direction: column; justify-content: center;
        transition: 0.3s;
    }
    .metric-container:hover { border-color: #6366F1; }
    .metric-label { font-size: 0.8em; color: #888; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5); color: white;
    }
    .profile-tag { 
        background: rgba(99, 102, 241, 0.1); color: #a5b4fc; padding: 3px 10px; 
        border-radius: 6px; font-size: 0.75em; margin: 2px; display: inline-block; border: 1px solid rgba(99, 102, 241, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEUR DE TRAITEMENT ---

def apply_perceptual_filter(y, sr):
    nyq = 0.5 * sr
    low, high = 100 / nyq, 3000 / nyq 
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_enhanced_chroma(y, sr, tuning):
    y_harm = librosa.effects.harmonic(y, margin=8.0)
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, sr=sr, tuning=tuning, 
        n_chroma=12, bins_per_octave=36, 
        fmin=librosa.note_to_hz('C2')
    )
    chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
    return np.power(chroma, 3.0) 

def solve_key_logic(chroma_vector):
    best_score, best_key = -1, ""
    winners = {}
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)

    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                note_str = f"{NOTES_LIST[i]} {mode}"
                if score > p_max:
                    p_max, p_note = score, note_str
                
                total_score = score * 1.25 if p_name == "bellman" else score
                if total_score > best_score:
                    best_score, best_key = total_score, note_str
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "details": winners}

def get_camelot(key_str):
    try:
        n, m = key_str.split(" ")
        return BASE_CAMELOT_MINOR.get(n, "??") if m == 'minor' else BASE_CAMELOT_MAJOR.get(n, "??")
    except: return "??"

def play_chord_button(note_mode, uid):
    if not note_mode or " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s").replace("-","_")
    return components.html(f"""
    <div style="height:100%; display:flex; align-items:center;">
    <button id="{js_id}" style="background:linear-gradient(90deg, #6366F1, #8B5CF6); color:white; border:none; border-radius:12px; padding:15px; cursor:pointer; font-weight:bold; width:100%; font-family: sans-serif; box-shadow: 0 4px 15px rgba(99,102,241,0.4);">
        🔊 TESTER {n} {m.upper()}
    </button>
    </div>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    document.getElementById('{js_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{m}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(it => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, it/12), now);
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.2, now+0.1); g.gain.exponentialRampToValueAtTime(0.01, now+1.5);
            o.connect(g); g.connect(ctx.destination); o.start(now); o.stop(now+1.5);
        }});
    }};
    </script>""", height=110)

# --- ANALYSE PRINCIPALE ---

@st.cache_data(show_spinner=False)
def process_audio(file_bytes, file_name):
    try:
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
        y_harm, y_perc = librosa.effects.hpss(y, margin=(1.5, 5.0))
        
        tuning = librosa.estimate_tuning(y=y_harm, sr=sr)
        duration = librosa.get_duration(y=y_harm, sr=sr)
        y_filt = apply_perceptual_filter(y_harm, sr)
        
        step, timeline = 8, []
        votes = Counter()
        
        for start in range(0, int(duration) - step, step):
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            if np.max(np.abs(y_seg)) < 0.015: continue 
            
            chroma = get_enhanced_chroma(y_seg, sr, tuning)
            res = solve_key_logic(np.mean(chroma, axis=1))
            
            weight = (res['score'] ** 3) * 100
            votes[res['key']] += weight
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        if not timeline: return {"error": "Audio trop court ou silencieux"}

        top_two = votes.most_common(2)
        final_key = top_two[0][0]
        
        full_chroma = get_enhanced_chroma(y_harm, sr, tuning)
        final_details_res = solve_key_logic(np.mean(full_chroma, axis=1))

        if len(top_two) > 1:
            margin = (top_two[0][1] - top_two[1][1]) / top_two[0][1]
            if margin < 0.15:
                final_key = final_details_res['details']['bellman']

        avg_conf = int(pd.DataFrame(timeline)[pd.DataFrame(timeline)['Note'] == final_key]['Conf'].mean())
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # --- ENRICHISSEMENT GRAPHIQUE POUR TELEGRAM ---
        df_tl = pd.DataFrame(timeline)
        fig = px.line(df_tl, x="Temps", y="Note", markers=True, 
                      category_orders={"Note": NOTES_ORDER}, 
                      template="plotly_dark",
                      title=f"Stabilité: {file_name}")
        
        fig.update_layout(
            paper_bgcolor='#0e1117', 
            plot_bgcolor='rgba(0,0,0,0)', 
            margin=dict(l=50, r=20, t=60, b=50),
            xaxis=dict(gridcolor='#1e293b', title="Temps (sec)"),
            yaxis=dict(gridcolor='#1e293b', title="Note / Tonalité"),
            font=dict(color="white")
        )
        img_bytes = fig.to_image(format="png", width=1200, height=600)

        output = {
            "name": file_name, "tempo": int(float(tempo)), "tuning": round(tuning, 2),
            "key": final_key, "camelot": get_camelot(final_key), "conf": avg_conf,
            "details": final_details_res['details'], "timeline": timeline, "plot": img_bytes
        }
        del y, y_harm, y_perc, y_filt, full_chroma; gc.collect()
        return output
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE ---

st.title("🎧 RCDJ228 M3 - VERSION PRO FINALE")
uploaded_files = st.file_uploader("📂 Chargez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    pbar = st.progress(0)
    for i, f in enumerate(uploaded_files):
        file_data = f.read()
        res = process_audio(file_data, f.name)
        
        if "error" in res:
            st.error(f"Erreur : {res['error']}")
            continue

        with st.expander(f"📊 ANALYSE : {res['name']}", expanded=True):
            bg_grad = "linear-gradient(135deg, #1e3a8a, #581c87)" if res['conf'] > 70 else "linear-gradient(135deg, #334155, #0f172a)"
            st.markdown(f"""
                <div class="final-decision-box" style="background:{bg_grad};">
                    <p style="margin:0; opacity:0.8; letter-spacing:3px; font-weight:300;">TONALITÉ DÉTECTÉE</p>
                    <h1 style="font-size:5.5em; margin:10px 0; font-weight:900;">{res['key']}</h1>
                    <p style="margin:0; font-size:1.5em;">CAMELOT: {res['camelot']} | CONFIANCE: {res['conf']}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns([1, 1, 1])
            with c1: 
                st.markdown(f'<div class="metric-container"><div class="metric-label">Tempo</div><div class="value-custom">{res["tempo"]} BPM</div></div>', unsafe_allow_html=True)
            with c2: play_chord_button(res['key'], f.name)
            with c3: 
                tags_html = "".join([f<span class='profile-tag'>{p}: {v}</span>" for p, v in res['details'].items()])
                st.markdown(f'<div class="metric-container"><div class="metric-label">Stabilité Algorithmique</div><div>{tags_html}</div></div>', unsafe_allow_html=True)
            
            st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", markers=True, category_orders={"Note": NOTES_ORDER}, template="plotly_dark"), use_container_width=True)

            # --- RAPPORT TELEGRAM ENRICHI ---
            try:
                caption = (
                    f"🎧 *RAPPORT D'ANALYSE RCDJ228 PRO*\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"📁 *Fichier:* `{res['name']}`\n"
                    f"🎹 *Key Principale:* `{res['key']}`\n"
                    f"🎼 *Camelot:* `{res['camelot']}`\n"
                    f"⏱ *Tempo:* `{res['tempo']} BPM`\n"
                    f"🎯 *Confiance:* `{res['conf']}%`\n"
                    f"🎸 *Tuning:* `{res['tuning']} Hz`\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"🔬 *Détails Algorithmes:*\n"
                    f"• Bellman: `{res['details']['bellman']}`\n"
                    f"• Krumhansl: `{res['details']['krumhansl']}`\n"
                    f"• Temperley: `{res['details']['temperley']}`\n"
                    f"━━━━━━━━━━━━━━━━━━━━\n"
                    f"⚡ _Analyse effectuée par M3 Pro Engine_"
                )
                
                requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                              files={'photo': res['plot']}, 
                              data={'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'})
            except Exception as e:
                st.warning(f"Telegram non envoyé : {e}")

        pbar.progress((i + 1) / len(uploaded_files))

if st.sidebar.button("🧹 Nettoyer le cache"):
    st.cache_data.clear()
    st.rerun()
