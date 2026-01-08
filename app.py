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
# Harmonisation des notations (Flat vs Sharp) pour la correspondance Camelot
BASE_CAMELOT_MINOR = {
    'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A',
    'G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'
}
BASE_CAMELOT_MAJOR = {
    'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B',
    'D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'
}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Profils de tonalité améliorés
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
    """Calcule la tonalité la plus probable sans biais artificiel."""
    best_score, best_key, winners = -1.0, "", {}
    # Normalisation Min-Max du vecteur de chroma
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1.0, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                # Corrélation de Pearson entre le chroma et le profil décalé
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > p_max: 
                    p_max, p_note = score, f"{NOTES_LIST[i]} {mode}"
                
                # Système de vote global (on retire le multiplicateur 1.1 pour éviter les erreurs Db/D#m)
                if score > best_score: 
                    best_score, best_key = score, f"{NOTES_LIST[i]} {mode}"
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "details": winners}

def process_audio(file_buffer, file_name, progress_bar, status_text):
    try:
        # 1. Chargement du fichier
        audio_bytes = file_buffer.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # 2. Séparation Harmonique / Percussive (HPSS) 
        # Crucial pour ne pas laisser les kicks fausser la détection des notes
        status_text.text(f"Séparation harmonique de {file_name}...")
        y_harmonic = librosa.effects.harmonic(y)
        
        # 3. Estimation Tuning
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        # 4. Segmentation et Analyse
        step_sec, hop_sec = 8, 4
        votes = Counter()
        timeline, all_chromas = [], []
        
        step_samples = step_sec * sr
        hop_samples = hop_sec * sr
        
        status_text.text(f"Analyse spectrale de {file_name}...")
        for start_sample in range(0, len(y_harmonic) - step_samples, hop_samples):
            y_seg = y_harmonic[start_sample : start_sample + step_samples]
            
            # Augmentation de bins_per_octave pour une meilleure résolution (24 au lieu de 12)
            chroma = librosa.feature.chroma_cqt(
                y=y_seg, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24
            )
            mean_chroma_seg = np.mean(chroma, axis=1)
            all_chromas.append(mean_chroma_seg)
            
            res = solve_key_logic(mean_chroma_seg)
            # Pondération par le carré du score pour favoriser les segments très clairs
            votes[res['key']] += (res['score'] ** 2)
            
            current_time = start_sample / sr
            timeline.append({"Temps": current_time, "Note": res['key'], "Conf": round(res['score']*100, 1)})
            
            progress = min(start_sample / len(y_harmonic), 1.0)
            progress_bar.progress(progress)

        if not votes:
            return {"error": "Signal trop faible"}

        # Extraction des résultats finaux
        final_key = votes.most_common(1)[0][0]
        full_chroma_avg = np.mean(all_chromas, axis=0)
        
        # 5. Calcul du Tempo (sur le signal original pour mieux détecter les transitoires)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

        # 6. Conversion Camelot
        key_root = final_key.split(' ')[0]
        camelot_code = (BASE_CAMELOT_MINOR if 'minor' in final_key else BASE_CAMELOT_MAJOR).get(key_root, "??")

        return {
            "name": file_name, 
            "tempo": int(np.round(tempo)), 
            "key": final_key,
            "camelot": camelot_code,
            "conf": int(pd.DataFrame(timeline)['Conf'].mean()),
            "chroma_vals": full_chroma_avg,
            "timeline": timeline
        }
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine")
st.markdown("---")

uploaded_files = st.file_uploader("📂 Glissez vos fichiers audio (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        pbar = st.progress(0)
        stext = st.empty()
        
        # Analyse
        res = process_audio(f, f.name, pbar, stext)
        pbar.empty()
        stext.empty()

        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}")
            continue

        # Affichage des résultats
        with st.expander(f"💎 ANALYSE TERMINÉE : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown(f"""
                    <div style="background:linear-gradient(135deg, #1e3a8a, #581c87); padding:30px; border-radius:15px; text-align:center; color:white; border: 1px solid #ffffff33;">
                        <small style="opacity:0.8; letter-spacing: 2px;">TONALITÉ DÉTECTÉE</small>
                        <h1 style="font-size:3.5rem; margin:10px 0;">{res['key'].upper()}</h1>
                        <div style="display:flex; justify-content:center; gap:20px; font-weight:bold;">
                            <span>CAMELOT: {res['camelot']}</span>
                            <span>|</span>
                            <span>FIABILITÉ: {res['conf']}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                st.metric("Tempo estimé", f"{res['tempo']} BPM")

            with col2:
                # Graphique Radar des fréquences
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=res['chroma_vals'], 
                    theta=NOTES_LIST, 
                    fill='toself', 
                    line_color='#00FFAA',
                    name="Empreinte Harmonique"
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False)), 
                    height=300, 
                    margin=dict(l=40,r=40,t=40,b=40), 
                    template="plotly_dark",
                    title="Analyse des Notes"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            # Timeline de stabilité
            df_tl = pd.DataFrame(res['timeline'])
            fig_line = px.line(df_tl, x="Temps", y="Note", title="Stabilité Harmonique au cours du temps", template="plotly_dark")
            fig_line.update_traces(line_color='#636EFA')
            st.plotly_chart(fig_line, use_container_width=True)
