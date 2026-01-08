import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import requests
import io # Ajouté pour la gestion des buffers
import streamlit.components.v1 as components

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine", page_icon="🎧", layout="wide")

# --- CONSTANTES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

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

def solve_key_logic(chroma_vector):
    best_score, best_key, winners = -1.0, "", {}
    # Normalisation
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1.0, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > p_max: 
                    p_max, p_note = score, f"{NOTES_LIST[i]} {mode}"
                
                t_score = score * 1.1 if p_name == "krumhansl" else score
                if t_score > best_score: 
                    best_score, best_key = t_score, f"{NOTES_LIST[i]} {mode}"
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "details": winners}

def process_audio(file_buffer, file_name, progress_bar, status_text):
    try:
        # 1. Charger l'audio une seule fois en mémoire pour éviter les problèmes de curseur
        # On utilise io.BytesIO pour copier le buffer au cas où
        audio_bytes = file_buffer.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 2. Estimation Tuning
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        
        # 3. Segmentation
        step_sec = 8
        hop_sec = 4
        votes = Counter()
        timeline = []
        all_chromas = []

        # Découpage en indices d'échantillons
        step_samples = step_sec * sr
        hop_samples = hop_sec * sr
        
        for start_sample in range(0, len(y) - step_samples, hop_samples):
            y_seg = y[start_sample : start_sample + step_samples]
            
            # Analyse Chroma CQT
            chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning, n_chroma=12)
            mean_chroma_seg = np.mean(chroma, axis=1)
            all_chromas.append(mean_chroma_seg)
            
            res = solve_key_logic(mean_chroma_seg)
            votes[res['key']] += (res['score'] ** 2)
            
            current_time = start_sample / sr
            timeline.append({"Temps": current_time, "Note": res['key'], "Conf": round(res['score']*100, 1)})
            
            # Update progress
            progress = min(start_sample / len(y), 1.0)
            progress_bar.progress(progress)

        if not votes:
            return {"error": "Signal trop faible ou fichier illisible"}

        final_key = votes.most_common(1)[0][0]
        full_chroma_avg = np.mean(all_chromas, axis=0)
        
        # 4. Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        camelot_key = final_key.split(' ')[0]
        camelot_code = (BASE_CAMELOT_MINOR if 'minor' in final_key else BASE_CAMELOT_MAJOR).get(camelot_key, "??")

        return {
            "name": file_name, "tempo": int(float(tempo)), "key": final_key,
            "camelot": camelot_code,
            "conf": int(pd.DataFrame(timeline)['Conf'].mean()),
            "details": solve_key_logic(full_chroma_avg)['details'],
            "timeline": timeline,
            "chroma_vals": full_chroma_avg
        }
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine")

uploaded_files = st.file_uploader("📂 Glissez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        pbar = st.progress(0)
        stext = st.empty()
        stext.text(f"Analyse de {f.name}...")
        
        res = process_audio(f, f.name, pbar, stext)
        pbar.empty()
        stext.empty()

        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}")
            continue

        with st.expander(f"💎 RÉSULTATS : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                    <div style="background:linear-gradient(135deg, #1e3a8a, #581c87); padding:20px; border-radius:15px; text-align:center; color:white;">
                        <small>TONALITÉ</small>
                        <h1 style="font-size:3rem; margin:0;">{res['key'].upper()}</h1>
                        <p>CAMELOT: <b>{res['camelot']}</b> | {res['conf']}% Fiabilité</p>
                    </div>
                """, unsafe_allow_html=True)
                st.metric("Tempo", f"{res['tempo']} BPM")

            with col2:
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=res['chroma_vals'], theta=NOTES_LIST, fill='toself', line_color='#00FFAA'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=250, margin=dict(l=20,r=20,t=40,b=20), template="plotly_dark")
                st.plotly_chart(fig_radar, use_container_width=True)

            st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", title="Stabilité Harmonique", template="plotly_dark"), use_container_width=True)
