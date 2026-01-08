import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228's Ear - Pro Edition", page_icon="🎧", layout="wide")

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

def refine_mode_by_tierce(chroma_vector, detected_key):
    root_note = detected_key.split(' ')[0]
    root_idx = NOTES_LIST.index(root_note)
    minor_third_idx = (root_idx + 3) % 12
    major_third_idx = (root_idx + 4) % 12
    e_minor = chroma_vector[minor_third_idx]
    e_major = chroma_vector[major_third_idx]
    
    if "major" in detected_key and e_minor > (e_major * 1.2):
        return f"{root_note} minor"
    elif "minor" in detected_key and e_major > (e_minor * 1.2):
        return f"{root_note} major"
    return detected_key

def solve_key_logic(chroma_vector):
    cv = np.power(chroma_vector, 2.5) 
    cv /= (np.max(cv) + 1e-6)
    best_score, best_key = -1.0, ""
    
    for p_name in ["temperley", "shaath"]:
        p_data = PROFILES[p_name]
        for mode in ["major", "minor"]:
            for i in range(12):
                profile = np.roll(p_data[mode], i)
                score = np.corrcoef(cv, profile)[0, 1]
                tonique_boost = 1 + (0.25 * cv[i])
                quinte_idx = (i + 7) % 12
                quinte_boost = 1 + (0.15 * cv[quinte_idx])
                final_score = score * tonique_boost * quinte_boost
                if final_score > best_score: 
                    best_score, best_key = final_score, f"{NOTES_LIST[i]} {mode}"
                    
    best_key = refine_mode_by_tierce(cv, best_key)
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
        
        # 1. HPSS
        status_text.text("Extraction harmonique...")
        y_harmonic = librosa.effects.harmonic(y, margin=8.0)
        
        # 2. Pre-emphasis
        y_pre = librosa.effects.preemphasis(y_harmonic)
        
        # 3. Tuning : Calibration de précision (Standard 440Hz)
        status_text.text("Calcul du micro-tuning...")
        tuning_offset = librosa.estimate_tuning(y=y_pre, sr=sr)
        # Conversion en Cents pour l'utilisateur (1 demi-ton = 100 cents)
        tuning_cents = int(tuning_offset * 100)
        
        # 4. Analyse CQT corrigée par le tuning détecté
        status_text.text("Analyse spectrale haute résolution...")
        chroma = librosa.feature.chroma_cqt(y=y_pre, sr=sr, tuning=tuning_offset, 
                                            n_chroma=12, bins_per_octave=24)
        
        # 5. Lissage
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 41))
        full_chroma_avg = np.mean(chroma_smooth, axis=1)
        
        # 6. Détection Clé + BPM
        res_logic = solve_key_logic(full_chroma_avg)
        status_text.text("Tracking du tempo...")
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        key_root = res_logic['key'].split(' ')[0]
        camelot = (BASE_CAMELOT_MINOR if 'minor' in res_logic['key'] else BASE_CAMELOT_MAJOR).get(key_root, "??")

        return {
            "name": file_name, "tempo": int(np.round(tempo)), 
            "key": res_logic['key'], "camelot": camelot, 
            "chroma_vals": full_chroma_avg, "fiabilite": int(res_logic['score'] * 100), 
            "tuning_cents": tuning_cents,
            "sr": sr
        }
    except Exception as e:
        return {"name": file_name, "error": str(e)}

# --- INTERFACE ---
st.title("🎧 RCDJ228's Ear - Studio Analyzer")
st.markdown("##### Moteur Neural avec Correction de Diapason")

uploaded_files = st.file_uploader("📂 Chargez vos morceaux (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

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

        with st.expander(f"💎 ANALYSE TERMINÉE : {res['name']}", expanded=True):
            col1, col2 = st.columns([1, 1.2])
            
            with col1:
                color_code = "#10b981" if res['fiabilite'] > 75 else "#f59e0b"
                st.markdown(f"""
                    <div style="background:#0f172a; padding:25px; border-radius:15px; border-left: 10px solid {color_code}; color:white;">
                        <small style="opacity:0.7; text-transform:uppercase;">Tonalité Détectée (Standard 440Hz)</small>
                        <h1 style="margin:0; color:{color_code}; font-size:48px;">{res['key'].upper()}</h1>
                        <div style="display:flex; justify-content:space-between; margin-top:15px; font-weight:bold; border-top:1px solid #334155; padding-top:10px;">
                            <span>SYSTÈME : {res['camelot']}</span>
                            <span>TEMPO : {res['tempo']} BPM</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # --- SECTION TUNING PRO ---
                st.write("---")
                st.write("### 🛠 Correction du Tuning")
                
                t_cents = res['tuning_cents']
                if abs(t_cents) <= 3:
                    st.success("✅ **Accordage Parfait** (La 440Hz)")
                else:
                    direction = "Trop haut (Sharp)" if t_cents > 0 else "Trop bas (Flat)"
                    # La solution pro : on propose d'appliquer l'inverse pour corriger
                    correction = -t_cents 
                    st.warning(f"⚠️ **Désaccordé : {abs(t_cents)} cents** ({direction})")
                    st.info(f"👉 **Solution Pro :** Appliquez un Pitch Shift de **{correction} cents** dans votre DAW pour revenir au standard 440Hz.")

                st.write("### 🎹 Test de Concordance")
                chord_audio = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord_audio, sample_rate=res['sr'])
                st.caption("Si l'accord et le morceau 'battent' ou sonnent faux ensemble, utilisez la correction ci-dessus.")

            with col2:
                st.write("### Empreinte Harmonique")
                r_data = np.append(res['chroma_vals'], res['chroma_vals'][0])
                theta_data = NOTES_LIST + [NOTES_LIST[0]]
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=r_data, theta=theta_data, fill='toself', line_color=color_code,
                    fillcolor=f'rgba({int(color_code[1:3],16)}, {int(color_code[3:5],16)}, {int(color_code[5:7],16)}, 0.25)'
                ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), template="plotly_dark", height=400)
                st.plotly_chart(fig_radar, use_container_width=True)
                st.metric("Indice de Pureté", f"{res['fiabilite']}%")

st.caption("Moteur v3.6 Pro | Détection d'offset en Cents intégrée | 440Hz Reference.")
