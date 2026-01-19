import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import gc
import scipy.ndimage
from scipy.signal import butter, lfilter
import streamlit.components.v1 as components
import requests

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="DJ's Ear Pro Elite v3", page_icon="🎧", layout="wide")

# --- GESTION DES SECRETS (GitHub / Streamlit Cloud) ---
# Ces valeurs doivent être configurées dans le dashboard Streamlit Cloud (Settings > Secrets)
# ou en local dans .streamlit/secrets.toml
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES AVANCÉS ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

HYBRID_PROFILES = {
    "major": np.array([6.35, 2.30, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
}

# --- FONCTION D'ENVOI TELEGRAM ---
def send_telegram_expert(data, fig_timeline, fig_radar):
    """Envoie un rapport complet vers Telegram. Utilise les secrets GitHub/Streamlit."""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        st.warning("⚠️ Telegram non configuré (Secrets manquants)")
        return

    msg = (f"🎼 *DJ'S EAR PRO ELITE REPORT*\n"
           f"━━━━━━━━━━━━━━━━━━━━\n"
           f"📂 *Fichier:* `{data['name']}`\n\n"
           f"✅ *TONALITÉ PRINCIPALE*\n"
           f"└ Note : `{data['key'].upper()}`\n"
           f"└ Camelot : `{data['camelot']}`\n\n"
           f"📊 *MÉTRIQUES*\n"
           f"└ Tempo : `{data['tempo']} BPM`\n"
           f"└ Tuning : `{data['tuning']} Hz`\n"
           f"━━━━━━━━━━━━━━━━━━━━")

    try:
        # 1. Message Texte
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                     json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        
        # 2. Graphiques (Nécessite 'kaleido')
        for fig, title in [(fig_timeline, "Flux Harmonique"), (fig_radar, "Signature Spectrale")]:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                         data={"chat_id": CHAT_ID, "caption": f"📊 {title} - {data['name']}"},
                         files={"photo": img_bytes})
                         
    except Exception as e:
        st.error(f"Erreur Telegram: {e}")

# --- MOTEUR DE TRAITEMENT (V3) ---
def apply_2026_filters(y, sr):
    y = librosa.effects.preemphasis(y)
    y_harm, _ = librosa.effects.hpss(y, margin=(10.0, 2.0))
    nyq = 0.5 * sr
    low, high = 100 / nyq, 3000 / nyq
    b, a = butter(6, [low, high], btype='band')
    return lfilter(b, a, y_harm)

def multi_chroma_fusion(y, sr, tuning):
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=72, n_octaves=7)
    cens = librosa.feature.chroma_cens(y=y, sr=sr)
    stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=8192)
    fused = (0.5 * cqt) + (0.3 * cens) + (0.2 * stft)
    return scipy.ndimage.median_filter(fused, size=(1, 15))

def analyze_engine_v3(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050)
    
    tuning = librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=72)
    y_clean = apply_2026_filters(y, sr)
    chroma_fused = multi_chroma_fusion(y_clean, sr, tuning)
    duration = librosa.get_duration(y=y, sr=sr)
    
    steps = np.linspace(0, chroma_fused.shape[1], 40, dtype=int)
    results_stream = []
    
    for i in range(len(steps)-1):
        segment = chroma_fused[:, steps[i]:steps[i+1]]
        avg_chroma = np.mean(segment, axis=1)
        best_score = -1
        best_key = "Ambiguous"
        
        for mode in ["major", "minor"]:
            for n in range(12):
                ref = np.roll(HYBRID_PROFILES[mode], n)
                score = np.corrcoef(avg_chroma, ref)[0, 1]
                if score > best_score:
                    best_score = score
                    best_key = f"{NOTES[n]} {mode}"
        
        results_stream.append({"time": (steps[i]/chroma_fused.shape[1])*duration, "key": best_key, "score": best_score})

    keys_found = [r['key'] for r in results_stream]
    main_key = Counter(keys_found).most_common(1)[0][0]
    
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)

    return {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "tempo": int(float(tempo)), "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": results_stream, "name": file_name,
        "chroma_avg": np.mean(chroma_fused, axis=1)
    }

# --- INTERFACE ---
st.title("🎧 DJ's Ear Elite v3 (Fusion Engine)")

# Sidebar pour vérifier la config
with st.sidebar:
    st.header("⚙️ Configuration")
    if TELEGRAM_TOKEN and CHAT_ID:
        st.success("Telegram Secret : OK")
    else:
        st.error("Telegram Secret : MANQUANT")
    
    if st.button("Reset Cache"):
        st.cache_data.clear()
        st.rerun()

files = st.file_uploader("Upload Audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse Deep Fusion : {f.name}"):
            data = analyze_engine_v3(f.read(), f.name)
            
        with st.expander(f"📊 {data['name']}", expanded=True):
            col1, col2 = st.columns([1, 2])
            
        with col1:
    st.markdown(f"""
        <div style="background:#1e293b; padding:20px; border-radius:15px; border-left: 5px solid #3b82f6;">
            <h2 style="color:#60a5fa; margin:0;">{data['key'].upper()}</h2>
            <h1 style="font-size:3em; margin:0; color: white;">{data['camelot']}</h1>  <p style="opacity:0.7; color: white;">{data['tempo']} BPM | {data['tuning']} Hz</p> </div>
    """, unsafe_allow_html=True)
                
                fig_polar = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES, fill='toself', line_color='#60a5fa'))
                fig_polar.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_polar, use_container_width=True)

            with col2:
                df_timeline = pd.DataFrame(data['timeline'])
                fig_line = px.line(df_timeline, x="time", y="key", title="Stabilité Harmonique (Viterbi Flow)",
                                    markers=True, template="plotly_dark", color_discrete_sequence=["#3b82f6"])
                st.plotly_chart(fig_line, use_container_width=True)

            # --- ENVOI AUTOMATIQUE TELEGRAM ---
            if TELEGRAM_TOKEN and CHAT_ID:
                send_telegram_expert(data, fig_line, fig_polar)
                st.toast(f"✅ Rapport envoyé pour {data['name']}", icon="📲")
