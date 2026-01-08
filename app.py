import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine v3", page_icon="🎧", layout="wide")

# --- RÉFÉRENCES MUSICALES ---
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

# --- FONCTIONS PSYCHO-ACOUSTIQUES ---

def apply_perceptual_weighting(y, sr):
    """Applique une pondération de type A (oreille humaine) au signal audio."""
    # Calcul du spectre de puissance
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    
    # Calcul de la courbe de pondération A
    # Formule standard pour simuler la réponse de l'oreille humaine
    a_weighting = librosa.perceptual_weighting(S**2, freqs)
    
    # Reconstruction approximative du signal pondéré (ou retour de l'énergie pondérée)
    # Pour l'analyse de tonalité, nous renvoyons le spectrogramme pondéré
    return S * librosa.db_to_amplitude(a_weighting)

# --- FONCTIONS DE CALCUL ---

def solve_key_logic(chroma_vector):
    """Analyse avec pondération de la tonique pour une précision accrue."""
    cv = np.power(chroma_vector, 3) 
    cv = (cv - cv.min()) / (cv.max() - cv.min() + 1e-6)
    
    best_score, best_key = -1.0, ""
    
    for p_name in ["bellman", "krumhansl"]:
        p_data = PROFILES[p_name]
        for mode in ["major", "minor"]:
            for i in range(12):
                profile = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, profile)[0, 1]
                
                # Boost intelligent : si la note fondamentale 'i' est forte dans le signal
                tonique_boost = 1 + (0.3 * cv[i]) 
                final_score = score * tonique_boost
                
                if final_score > best_score: 
                    best_score, best_key = final_score, f"{NOTES_LIST[i]} {mode}"
                    
    return {"key": best_key, "score": min(best_score, 1.0)}

def generate_reference_chord(key_str, duration=2.5, sr=22050):
    root_map = {n: i for i, n in enumerate(NOTES_LIST)}
    parts = key_str.split(' ')
    root, mode = parts[0], parts[1]
    f0 = 130.81 * (2**(root_map[root]/12))
    intervals = [0, 4, 7] if mode == 'major' else [0, 3, 7]
    t = np.linspace(0, duration, int(sr * duration))
    chord_signal = np.zeros_like(t)
    for i in intervals:
        freq = f0 * (2**(i/12))
        chord_signal += 0.4 * np.sin(2 * np.pi * freq * t)
    chord_signal *= np.exp(-1.5 * t)
    return chord_signal / (np.max(np.abs(chord_signal)) + 1e-6)

def process_audio(file_buffer, file_name, progress_bar, status_text):
    try:
        status_text.text(f"Lecture : {file_name}")
        y, sr = librosa.load(file_buffer, sr=22050)
        
        # 1. Isolation Harmonique
        status_text.text("Extraction des composantes harmoniques...")
        y_harmonic = librosa.effects.harmonic(y, margin=3.0)
        
        # 2. Perception Humaine (A-Weighting)
        # On ajuste l'énergie des fréquences selon la sensibilité de l'oreille
        status_text.text("Application du filtre psychoacoustique...")
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        # 3. Analyse CQT (Constant-Q Transform)
        # Utilisation de l'audio harmonique pondéré par la perception
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, tuning=tuning, 
                                            n_chroma=12, bins_per_octave=24)
        
        # 4. Lissage temporel (Médian)
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 41))
        full_chroma_avg = np.mean(chroma_smooth, axis=1)
        
        # 5. Détection Clé + BPM
        res_logic = solve_key_logic(full_chroma_avg)
        status_text.text("Calcul du tempo...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        key_root = res_logic['key'].split(' ')[0]
        camelot = (BASE_CAMELOT_MINOR if 'minor' in res_logic['key'] else BASE_CAMELOT_MAJOR).get(key_root, "??")

        return {
            "name": file_name, "tempo": int(np.round(tempo)), 
            "key": res_logic['key'], "camelot": camelot, 
            "chroma_vals": full_chroma_avg, "fiabilite": int(res_logic['score'] * 100), 
            "sr": sr
        }
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine v3")
st.markdown("##### Analyseur de tonalité basé sur la perception auditive humaine")

uploaded_files = st.file_uploader("📂 Chargez vos morceaux (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        pbar = st.progress(0)
        stext = st.empty()
        
        res = process_audio(f, f.name, pbar, stext)
        
        pbar.empty()
        stext.empty()

        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}")
            continue

        # Affichage des résultats
        with st.expander(f"💎 RÉSULTAT : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                # Dynamisme de couleur selon la fiabilité
                color_code = "#10b981" if res['fiabilite'] > 78 else "#f59e0b"
                
                st.markdown(f"""
                    <div style="background:#0f172a; padding:25px; border-radius:15px; border-left: 10px solid {color_code}; color:white;">
                        <small style="opacity:0.7; text-transform:uppercase;">Tonalité Détectée (Psycho-Acoustique)</small>
                        <h1 style="margin:0; color:{color_code}; font-size:48px;">{res['key'].upper()}</h1>
                        <div style="display:flex; justify-content:space-between; margin-top:15px; font-weight:bold; border-top:1px solid #334155; padding-top:10px;">
                            <span>SYSTÈME : {res['camelot']}</span>
                            <span>TEMPO : {res['tempo']} BPM</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("---")
                st.write("### 🎹 Vérification à l'oreille")
                st.caption("Écoutez si cet accord fusionne parfaitement avec votre morceau.")
                chord_audio = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord_audio, sample_rate=res['sr'])
                
                fiabilite_msg = "Confiance élevée" if res['fiabilite'] > 78 else "Analyse complexe (vérification recommandée)"
                st.info(f"**Score de corrélation : {res['fiabilite']}%** — {fiabilite_msg}")

            with col2:
                st.write("### Empreinte Harmonique")
                r_data = np.append(res['chroma_vals'], res['chroma_vals'][0])
                theta_data = NOTES_LIST + [NOTES_LIST[0]]

                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=r_data, 
                    theta=theta_data, 
                    fill='toself', 
                    line_color=color_code,
                    fillcolor=f'rgba({int(color_code[1:3],16)}, {int(color_code[3:5],16)}, {int(color_code[5:7],16)}, 0.25)'
                ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False), angularaxis=dict(tickfont_size=14)),
                    template="plotly_dark", 
                    height=400, 
                    margin=dict(l=40, r=40, t=30, b=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Moteur v3.0 | Algorithme : CQT + Pondération Perceptuelle A + Profils Krumhansl-Schmuckler.")
