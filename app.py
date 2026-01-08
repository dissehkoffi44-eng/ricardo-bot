import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine", page_icon="🎧", layout="wide")

# --- CONSTANTES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

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

# --- FONCTIONS AUDIO ---

def generate_reference_chord(key_str, duration=3.0, sr=22050):
    """Génère un accord de piano synthétique (sinusoïdal) pour vérification."""
    root_map = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5, 'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}
    
    parts = key_str.split(' ')
    root = parts[0]
    mode = parts[1]
    
    # Fréquence de base (Octave 3)
    f0 = 130.81 * (2**(root_map[root]/12))
    
    # Intervalles (Fondamentale, Tierce, Quinte)
    intervals = [0, 4, 7] if mode == 'major' else [0, 3, 7]
    
    t = np.linspace(0, duration, int(sr * duration))
    chord_signal = np.zeros_like(t)
    
    for i in intervals:
        freq = f0 * (2**(i/12))
        # Ajout d'harmoniques pour un son plus riche que du pur sinus
        chord_signal += 0.5 * np.sin(2 * np.pi * freq * t)
        chord_signal += 0.2 * np.sin(2 * np.pi * 2 * freq * t) # Harmonique 1
        
    # Enveloppe simple pour éviter les clics
    envelope = np.exp(-t)
    chord_signal = chord_signal * envelope
    chord_signal /= np.max(np.abs(chord_signal)) # Normalisation
    
    return chord_signal

def solve_key_logic(chroma_vector):
    best_score, best_key = -1.0, ""
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > best_score: 
                    best_score, best_key = score, f"{NOTES_LIST[i]} {mode}"
    return {"key": best_key, "score": best_score}

def process_audio(file_buffer, file_name, progress_bar, status_text):
    try:
        audio_bytes = file_buffer.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # Nettoyage Harmonique (HPSS)
        status_text.text("Séparation des harmoniques...")
        y_harmonic = librosa.effects.harmonic(y)
        
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        step_sec, hop_sec = 8, 4
        votes = Counter()
        timeline, all_chromas = [], []
        step_samples = step_sec * sr
        hop_samples = hop_sec * sr
        
        status_text.text("Analyse spectrale...")
        for start_sample in range(0, len(y_harmonic) - step_samples, hop_samples):
            y_seg = y_harmonic[start_sample : start_sample + step_samples]
            chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24)
            mean_chroma_seg = np.mean(chroma, axis=1)
            all_chromas.append(mean_chroma_seg)
            
            res = solve_key_logic(mean_chroma_seg)
            votes[res['key']] += (res['score'] ** 2)
            
            progress = min(start_sample / len(y_harmonic), 1.0)
            progress_bar.progress(progress)

        final_key = votes.most_common(1)[0][0]
        full_chroma_avg = np.mean(all_chromas, axis=0)
        
        # Tempo
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        key_root = final_key.split(' ')[0]
        camelot_code = (BASE_CAMELOT_MINOR if 'minor' in final_key else BASE_CAMELOT_MAJOR).get(key_root, "??")

        return {
            "name": file_name, "tempo": int(np.round(tempo)), "key": final_key,
            "camelot": camelot_code, "chroma_vals": full_chroma_avg,
            "sr": sr
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
        res = process_audio(f, f.name, pbar, stext)
        pbar.empty()
        stext.empty()

        if "error" in res:
            st.error(f"Erreur: {res['error']}")
            continue

        with st.expander(f"💎 RÉSULTAT : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""
                    <div style="background:linear-gradient(135deg, #1e3a8a, #581c87); padding:30px; border-radius:15px; text-align:center; color:white;">
                        <small>TONALITÉ</small>
                        <h1 style="font-size:3.5rem; margin:0;">{res['key'].upper()}</h1>
                        <p>CAMELOT: <b>{res['camelot']}</b> | {res['tempo']} BPM</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- AJOUT DE LA NOTE TÉMOIN ---
                st.write("### 🎹 Note Témoin (Vérification)")
                chord_audio = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord_audio, sample_rate=res['sr'])
                st.caption("Écoutez si cet accord sonne juste avec votre morceau.")

            with col2:
                fig_radar = go.Figure(data=go.Scatterpolar(r=res['chroma_vals'], theta=NOTES_LIST, fill='toself', line_color='#00FFAA'))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=300, template="plotly_dark", margin=dict(l=40,r=40,t=40,b=40))
                st.plotly_chart(fig_radar, use_container_width=True)
