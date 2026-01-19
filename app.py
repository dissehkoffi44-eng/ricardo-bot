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
st.set_page_config(page_title="RCDJ228 SNIPER ELITE M3", page_icon="🎯", layout="wide")

# Secrets Telegram
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
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .report-card { 
        padding: 30px; border-radius: 25px; text-align: center; color: white; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.2); color: #f87171;
        padding: 12px; border-radius: 12px; border: 1px solid #ef4444;
        margin-top: 15px; font-weight: bold; font-family: monospace;
    }
    .metric-box {
        background: #161b22; border-radius: 15px; padding: 15px; text-align: center; border: 1px solid #30363d;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT (MÉLANGE CODE 1 & 2) ---

def apply_pro_filters(y, sr):
    # Séparation harmonique (Code 2)
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    # Filtre passe-bande chirurgical
    nyq = 0.5 * sr
    b, a = butter(4, [80/nyq, 4500/nyq], btype='band')
    return lfilter(b, a, y_harm)

def get_triad_indices(key_name):
    """Calcule les indices des notes de la triade (Fondamentale, Tierce, Quinte)"""
    note_root, mode = key_name.split()
    root_idx = NOTES_LIST.index(note_root)
    if mode == "major":
        return [root_idx, (root_idx + 4) % 12, (root_idx + 7) % 12]
    else:
        return [root_idx, (root_idx + 3) % 12, (root_idx + 7) % 12]

def triad_arbitration(key1, key2, global_chroma):
    """LOGIQUE CODE 2 : Compare les candidats avec le Top 4 des notes dominantes"""
    # On identifie les 4 notes les plus présentes dans tout le morceau
    top_4_real = np.argsort(global_chroma)[-4:]
    
    def score_key(k):
        triad = get_triad_indices(k)
        # Score = combien de notes de la triade théorique sont dans le top 4 réel
        return sum(1 for note in triad if note in top_4_real)

    s1 = score_key(key1)
    s2 = score_key(key2)

    if s2 > s1:
        return key2, key1, True # On inverse : la modulation est plus cohérente
    return key1, key2, False

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_pro_filters(y, sr)
    
    # Analyse par segments (Code 1)
    step, timeline, votes = 6, [], Counter()
    for start in range(0, int(duration) - step, 3):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.01: continue
        
        c_seg = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, n_chroma=12)
        c_avg = np.mean(c_seg, axis=1)
        
        # Scoring via Profils
        best_score, best_k = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(c_avg, np.roll(PROFILES["krumhansl"][mode], i))[0, 1]
                if score > best_score:
                    best_score = score
                    best_k = f"{NOTES_LIST[i]} {mode}"
        
        weight = 1.5 if (start < 15 or start > (duration - 15)) else 1.0
        votes[best_k] += int(best_score * 100 * weight)
        timeline.append({"Temps": start, "Note": best_k, "Conf": best_score})

    # Extraction des 2 meilleurs candidats
    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    
    # Init variables modulation
    target_key = None
    triad_fix = False
    
    # Calcul de la signature chromatique globale
    global_chroma = np.mean(librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning), axis=1)

    # --- ACTION CODE 2 : Si deux notes dominent (Modulation ou hésitation) ---
    if len(most_common) > 1:
        potential_mod = most_common[1][0]
        # Si la 2ème note représente au moins 20% des votes
        if (votes[potential_mod] / sum(votes.values())) > 0.20:
            main_key, target_key, triad_fix = triad_arbitration(main_key, potential_mod, global_chroma)
        else:
            target_key = None # Négligeable

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
            g.gain.linearRampToValueAtTime(0.1, ctx.currentTime + 0.1);
            g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
            o.connect(g); g.connect(ctx.destination);
            o.start(); o.stop(ctx.currentTime + 1.5);
        }});
    }};
    """

def send_telegram(data):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    mod_info = f"\n⚠️ MODULATION: {data['target_key']} ({data['target_camelot']})" if data['modulation'] else ""
    fix_info = "\n✅ VALIDATION TRIADIQUE APPLIQUÉE" if data['triad_validated'] else ""
    
    msg = (f"🎯 *SNIPER M3 REPORT*\n"
           f"📂 `{data['name']}`\n"
           f"🎹 TONALITÉ: *{data['key'].upper()}*\n"
           f"🌀 CAMELOT: `{data['camelot']}`\n"
           f"🔥 CONF: `{data['conf']}%`\n"
           f"{mod_info}{fix_info}\n"
           f"⏱ BPM: `{data['tempo']}` | `{data['tuning']}Hz`")
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                      json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
    except: pass

# --- INTERFACE ---
st.title("🎯 RCDJ228 SNIPER M3 ELITE")

files = st.file_uploader("📥 Charger des fichiers", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse Sniper de {f.name}..."):
            data = analyze_full_engine(f.read(), f.name)
        
        # Affichage du rapport
        bg = "linear-gradient(135deg, #064e3b, #065f46)" if data['conf'] > 85 else "linear-gradient(135deg, #1e1b4b, #1e293b)"
        st.markdown(f"""
            <div class="report-card" style="background:{bg};">
                <h2 style="margin:0; opacity:0.8;">{data['name']}</h2>
                <h1 style="font-size:5.5em; margin:15px 0;">{data['key'].upper()}</h1>
                <p style="font-size:1.8em;">CAMELOT: <b>{data['camelot']}</b> | CONFIANCE: <b>{data['conf']}%</b></p>
                {"<div class='modulation-alert'>⚠️ MODULATION DÉTECTÉE : " + data['target_key'].upper() + " (" + data['target_camelot'] + ")</div>" if data['modulation'] else ""}
                {"<p style='color:#10b981; margin-top:10px;'>✓ Validé par analyse des triades dominantes</p>" if data['triad_validated'] else ""}
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:1.8em;'>{data['tempo']} BPM</span></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='metric-box'><b>TUNING</b><br><span style='font-size:1.8em;'>{data['tuning']} Hz</span></div>", unsafe_allow_html=True)
        with c3:
            btn_id = f"btn_{hash(f.name)}"
            components.html(f"""<button id="{btn_id}" style="width:100%; height:85px; background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.1em;">🎹 JOUER L'ACCORD</button><script>{get_chord_js(btn_id, data['key'])}</script>""", height=100)

        # Graphiques
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.plotly_chart(px.line(pd.DataFrame(data['timeline']), x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER}, title="Flux Harmonique"), use_container_width=True)
        with col_r:
            fig_r = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#10b981'))
            fig_r.update_layout(template="plotly_dark", title="Empreinte Chromatique", polar=dict(radialaxis=dict(visible=False)))
            st.plotly_chart(fig_r, use_container_width=True)

        send_telegram(data)
        st.toast(f"Rapport Sniper envoyé pour {f.name}")
        st.write("---")

if st.sidebar.button("🧹 Vider la mémoire"):
    st.cache_data.clear(); st.rerun()
