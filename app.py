import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine v3.6", page_icon="🎧", layout="wide")

# --- RÉFÉRENCES MUSICALES & PROFILS IA ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PROFILES = {
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
    },
    "shaath": {
        "major": [6.6, 2.0, 3.5, 2.3, 4.6, 4.0, 2.5, 5.2, 2.4, 3.7, 2.3, 3.4],
        "minor": [6.5, 2.7, 3.5, 5.4, 2.6, 3.5, 2.5, 5.2, 4.0, 2.7, 4.3, 3.2]
    }
}

# --- FONCTIONS DE CALCUL ---

def solve_key_logic(chroma_vector):
    """Analyse renforcée contre les fausses toniques (Ex: confusion Tonique/Quinte)."""
    # Amplification cubique pour isoler les pics réels des harmoniques fantômes
    cv = np.power(chroma_vector, 3.0) 
    cv /= (np.max(cv) + 1e-6)
    
    best_score, best_key = -1.0, ""
    
    for p_name in ["temperley", "shaath"]:
        p_data = PROFILES[p_name]
        for mode in ["major", "minor"]:
            for i in range(12):
                profile = np.roll(p_data[mode], i)
                
                # Corrélation statistique de base
                base_corr = np.corrcoef(cv, profile)[0, 1]
                
                # --- LOGIQUE ANTI-ERREUR ---
                # Presence physique de la tonique supposée
                tonique_presence = cv[i]
                # Presence de sa quinte (i+7)
                quinte_idx = (i + 7) % 12
                quinte_presence = cv[quinte_idx]
                
                # Pénalité : Si la quinte est beaucoup plus forte que la tonique, 
                # il y a de fortes chances que ce soit la quinte qui soit prise pour une tonique.
                penalty = 1.0
                if quinte_presence > (tonique_presence * 1.4):
                    penalty = 0.8  
                
                # Score final pondéré par la solidité de la tonique
                # Un profil qui colle bien MAIS dont la tonique est absente sera déclassé.
                final_score = base_corr * (1 + tonique_presence) * penalty
                
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
        
        # 1. HPSS (Isolation Harmonique)
        status_text.text("Nettoyage spectral (HPSS)...")
        y_harmonic = librosa.effects.harmonic(y, margin=8.0)
        
        # 2. Pre-emphasis & Tuning
        y_tuned = librosa.effects.preemphasis(y_harmonic)
        status_text.text("Calibration du diapason...")
        tuning = librosa.estimate_tuning(y=y_tuned, sr=sr)
        
        # 3. CQT Haute Résolution (fmin bas pour capter les toniques en basse)
        status_text.text("Analyse des chromas...")
        chroma = librosa.feature.chroma_cqt(y=y_tuned, sr=sr, tuning=tuning, 
                                            n_chroma=12, bins_per_octave=24, fmin=librosa.note_to_hz('C1'))
        
        # 4. Lissage Temporel
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
        return {"name": file_name, "error": str(e)}

# --- INTERFACE UTILISATEUR ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine v3.6")
st.markdown("##### Moteur Neural-Like : Temperley Profiles + Harmonic-Structural Weighting")

uploaded_files = st.file_uploader("📂 Chargez vos morceaux", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    all_results = []
    
    for f in uploaded_files:
        pbar = st.progress(0)
        stext = st.empty()
        res = process_audio(f, f.name, pbar, stext)
        all_results.append(res)
        pbar.empty()
        stext.empty()

    for res in all_results[::-1]:
        if "error" in res:
            st.error(f"Erreur sur {res['name']}: {res['error']}")
            continue

        with st.expander(f"💎 RÉSULTAT : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                color_code = "#10b981" if res['fiabilite'] > 70 else "#f59e0b"
                
                st.markdown(f"""
                    <div style="background:#0f172a; padding:25px; border-radius:15px; border-left: 10px solid {color_code}; color:white;">
                        <small style="opacity:0.7; text-transform:uppercase;">Tonalité Validée (Filtre Tonique)</small>
                        <h1 style="margin:0; color:{color_code}; font-size:48px;">{res['key'].upper()}</h1>
                        <div style="display:flex; justify-content:space-between; margin-top:15px; font-weight:bold; border-top:1px solid #334155; padding-top:10px;">
                            <span>SYSTÈME : {res['camelot']}</span>
                            <span>TEMPO : {res['tempo']} BPM</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write("---")
                st.write("### 🎹 Vérification à l'oreille")
                chord_audio = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord_audio, sample_rate=res['sr'])
                
                fiabilite_msg = "Analyse ultra-précise (Tonique confirmée)" if res['fiabilite'] > 70 else "Structure complexe : Quinte dominante ?"
                st.info(f"**Indice de confiance : {res['fiabilite']}%** — {fiabilite_msg}")

            with col2:
                st.write("### Empreinte Harmonique (Nettoyée)")
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
                    template="plotly_dark", height=400, margin=dict(l=40, r=40, t=30, b=30)
                )
                st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")
st.caption("Moteur v3.6 | Structural Anti-Trap Engine | CQT + HPSS + Profils de Temperley.")
