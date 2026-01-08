import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine v3.7", page_icon="🎧", layout="wide")

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
    """Moteur Psycho-Acoustique : Analyse de stabilité tonale."""
    # Amplification non-linéaire (Simule la focalisation de l'attention humaine)
    cv = np.power(chroma_vector, 3.5) 
    cv /= (np.max(cv) + 1e-6)
    
    best_score, best_key = -1.0, ""
    
    for p_name in ["temperley", "shaath"]:
        p_data = PROFILES[p_name]
        for mode in ["major", "minor"]:
            for i in range(12):
                profile = np.roll(p_data[mode], i)
                corr = np.corrcoef(cv, profile)[0, 1]
                
                # --- LOGIQUE DE PERCEPTION ---
                tonique_pres = cv[i]
                quinte_pres = cv[(i + 7) % 12]
                tierce_idx = (i + 3) % 12 if mode == "minor" else (i + 4) % 12
                tierce_pres = cv[tierce_idx]
                
                # Un morceau sonne dans une clé si la triade (1-3-5) est stable
                stability = (tonique_pres * 1.8) + (quinte_pres * 1.2) + (tierce_pres * 0.9)
                
                # Pénalité : Si la quinte domine trop la tonique (Erreur classique de 5 semi-tons)
                if quinte_pres > (tonique_pres * 1.7):
                    stability *= 0.7
                
                final_score = corr * stability
                
                if final_score > best_score: 
                    best_score, best_key = final_score, f"{NOTES_LIST[i]} {mode}"
                    
    return {"key": best_key, "score": min(best_score / 2.5, 1.0)}

def generate_reference_chord(key_str, duration=2.5, sr=22050):
    """Génère un accord de référence pour validation à l'oreille."""
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
        status_text.text(f"Analyse perceptive : {file_name}")
        y, sr = librosa.load(file_buffer, sr=22050)
        
        # 1. PSYCHO-FILTRAGE
        y = librosa.effects.preemphasis(y)

        # 2. HPSS AGRESSIF (On dégage les kicks qui saturent le spectre)
        status_text.text("Séparation des composantes harmoniques...")
        y_harmonic = librosa.effects.harmonic(y, margin=10.0) 
        
        # 3. TUNING & CQT (Basses profondes incluses C1)
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        status_text.text("Extraction de l'empreinte tonale...")
        chroma = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, tuning=tuning, 
            n_chroma=12, bins_per_octave=24, fmin=librosa.note_to_hz('C1')
        )
        
        # 4. LISSAGE COGNITIF
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 61))
        full_chroma_avg = np.mean(chroma_smooth, axis=1)
        
        # 5. DÉTECTION
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

# --- INTERFACE ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine v3.7")
st.markdown("##### Intelligence Auditive : Simulation de la perception humaine (Tonique-Root Focus)")

uploaded_files = st.file_uploader("📂 Déposez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

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

        with st.expander(f"🎵 ANALYSE : {res['name']}", expanded=True):
            c1, c2 = st.columns([1, 1.2])
            with c1:
                color = "#10b981" if res['fiabilite'] > 65 else "#6366f1"
                st.markdown(f"""
                    <div style="background:#111827; padding:20px; border-radius:12px; border-left: 10px solid {color};">
                        <h4 style="margin:0; opacity:0.6; color:white;">CLÉ PERCEPTIVE</h4>
                        <h1 style="font-size:50px; color:{color}; margin:10px 0;">{res['key'].upper()}</h1>
                        <p style="font-size:20px; color:white;"><b>CAMELOT : {res['camelot']}</b> | <b>{res['tempo']} BPM</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**Confiance Auditive : {res['fiabilite']}%**")
                
                st.write("### 🎹 Vérification à l'oreille")
                chord = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord, sample_rate=res['sr'])

            with c2:
                st.write("### Empreinte Harmonique")
                r_vals = np.append(res['chroma_vals'], res['chroma_vals'][0])
                theta_vals = NOTES_LIST + [NOTES_LIST[0]]
                fig = go.Figure(go.Scatterpolar(
                    r=r_vals, theta=theta_vals, fill='toself', 
                    line_color=color, fillcolor=f'rgba(99, 102, 241, 0.2)'
                ))
                fig.update_layout(
                    template="plotly_dark", height=350, 
                    margin=dict(l=40,r=40,b=20,t=20),
                    polar=dict(radialaxis=dict(visible=False), angularaxis=dict(tickfont_size=14))
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Moteur v3.7 | Structural Anti-Trap Engine | Focus Tonique-Basse")
