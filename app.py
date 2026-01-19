import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import scipy.ndimage
from scipy.signal import butter, lfilter
import requests
import gc

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="DJ's Ear Pro Elite v3.1", page_icon="🚀", layout="wide")

# CONFIGURATION TELEGRAM (Via st.secrets ou Sidebar pour tests)
# Pour usage pro, créez un fichier .streamlit/secrets.toml avec :
# TELEGRAM_TOKEN = "votre_token"
# CHAT_ID = "votre_id"
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN") or st.sidebar.text_input("Bot Token Telegram", type="password")
CHAT_ID = st.secrets.get("CHAT_ID") or st.sidebar.text_input("Chat ID Telegram")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A',
    'No Key': '??'
}

HYBRID_PROFILES = {
    "major": np.array([6.35, 2.30, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
    "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
}

# --- FONCTION D'ENVOI TELEGRAM ---

def send_telegram_expert(data, fig_timeline, fig_radar):
    """Génère et envoie un rapport multimédia complet sur Telegram"""
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return False

    # Construction du message texte (Markdown)
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
        # 1. Envoi du texte
        base_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
        requests.post(f"{base_url}/sendMessage", json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})

        # 2. Envoi des graphiques (Convertis en PNG via Kaleido)
        for fig, label in [(fig_timeline, "FLUX_HARMONIQUE"), (fig_radar, "SIGNATURE_CHROMA")]:
            # engine="kaleido" est nécessaire pour transformer Plotly en PNG
            img_bytes = fig.to_image(format="png", engine="kaleido")
            requests.post(f"{base_url}/sendPhoto", 
                          data={"chat_id": CHAT_ID, "caption": f"📊 {label} - {data['name']}"},
                          files={"photo": img_bytes})
        return True
    except Exception as e:
        st.error(f"Erreur Telegram: {e}")
        return False

# --- LOGIQUE DE TRAITEMENT (VOTRE ORDRE CNN-CRF) ---

def apply_aggressive_preprocessing(y, sr):
    y = librosa.effects.preemphasis(y)
    y_harm, _ = librosa.effects.hpss(y, margin=(8.0, 2.0))
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 3000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def extract_fused_features(y, sr, tuning):
    cqt = librosa.feature.chroma_cqt(y=y, sr=sr, tuning=tuning, bins_per_octave=72)
    hpcp = librosa.feature.chroma_cens(y=y, sr=sr)
    stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=4096)
    return (0.5 * cqt) + (0.3 * hpcp) + (0.2 * stft)

def analyze_engine_v3(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050)
    
    duration = librosa.get_duration(y=y, sr=sr)
    tuning = librosa.estimate_tuning(y=y, sr=sr, bins_per_octave=72)
    y_clean = apply_aggressive_preprocessing(y, sr)
    features = extract_fused_features(y_clean, sr, tuning)
    
    # Génération Logits (24+1 classes)
    steps = np.linspace(0, features.shape[1], 80, dtype=int)
    logits_matrix = []
    labels_list = [f"{n} {m}" for m in ["major", "minor"] for n in NOTES]
    
    for i in range(len(steps)-1):
        seg = np.mean(features[:, steps[i]:steps[i+1]], axis=1)
        scores = [np.corrcoef(seg, np.roll(HYBRID_PROFILES[m], NOTES.index(n)))[0, 1] 
                  for m in ["major", "minor"] for n in NOTES]
        # 25ème classe (No Key) si confiance faible
        scores.append(1.0 if max(scores) < 0.35 else 0.0)
        logits_matrix.append(scores)
    
    # Viterbi / Median Filter
    clean_logits = scipy.ndimage.median_filter(np.array(logits_matrix), size=(9, 1))
    
    timeline = []
    for t in range(len(clean_logits)):
        idx = np.argmax(clean_logits[t])
        timeline.append({"time": (t/len(clean_logits))*duration, 
                         "key": (labels_list + ["No Key"])[idx], 
                         "score": float(clean_logits[t][idx])})

    # Vote Pondéré (Corps >> Intro/Outro)
    weights = np.hanning(len(timeline))
    votes = Counter()
    for i, ent in enumerate(timeline):
        if ent['key'] != "No Key": votes[ent['key']] += weights[i]
    
    main_key = votes.most_common(1)[0][0] if votes else "No Key"
    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)

    return {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "tempo": int(float(tempo)), "tuning": round(440 * (2**(tuning/12)), 1),
        "timeline": timeline, "name": file_name, "chroma_avg": np.mean(features, axis=1)
    }, y

# --- INTERFACE ---
st.title("🎧 DJ's Ear Pro Elite v3.1")
files = st.file_uploader("📂 Déposer vos morceaux", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    for f in reversed(files):
        with st.spinner(f"Analyse profonde : {f.name}"):
            data, _ = analyze_engine_v3(f.read(), f.name)
        
        with st.expander(f"✅ {data['name']} - {data['key']}", expanded=True):
            c1, c2 = st.columns([1, 2])
            
            with c1:
                st.markdown(f"### {data['camelot']} | {data['key']}")
                st.write(f"Tempo: {data['tempo']} BPM")
                # Radar Graph
                fig_r = go.Figure(data=go.Scatterpolar(r=data['chroma_avg'], theta=NOTES, fill='toself'))
                fig_r.update_layout(template="plotly_dark", title="Signature Chromatique")
                st.plotly_chart(fig_r, use_container_width=True)
                
            with c2:
                # Timeline Graph
                df_tl = pd.DataFrame(data['timeline'])
                fig_l = px.line(df_tl, x="time", y="key", markers=True, template="plotly_dark", title="Stabilité (Viterbi Cleaned)")
                st.plotly_chart(fig_l, use_container_width=True)

            # --- DÉCLENCHEMENT TELEGRAM ---
            if send_telegram_expert(data, fig_l, fig_r):
                st.toast(f"🚀 Rapport Telegram envoyé pour {f.name} !")
            
            gc.collect()
