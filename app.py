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

# Profils cognitifs améliorés
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
    """Moteur Psycho-Acoustique v3.7 : Analyse de stabilité tonale."""
    # 1. Amplification non-linéaire (Simule la focalisation de l'attention humaine)
    cv = np.power(chroma_vector, 3.5) 
    cv /= (np.max(cv) + 1e-6)
    
    best_score, best_key = -1.0, ""
    
    for p_name in ["temperley", "shaath"]:
        p_data = PROFILES[p_name]
        for mode in ["major", "minor"]:
            for i in range(12):
                profile = np.roll(p_data[mode], i)
                
                # Corrélation statistique
                corr = np.corrcoef(cv, profile)[0, 1]
                
                # --- ANALYSE DE LA TONIQUE (L'oreille privilégie la racine) ---
                tonique_pres = cv[i]
                quinte_pres = cv[(i + 7) % 12]
                tierce_pres = cv[(i + 3) % 12] if mode == "minor" else cv[(i + 4) % 12]
                
                # Un morceau "sonne" dans une clé si la tonique ET sa quinte sont cohérentes
                # On crée un multiplicateur de stabilité
                stability = (tonique_pres * 1.6) + (quinte_pres * 1.1) + (tierce_pres * 0.8)
                
                # Pénalité de "Quinte Perdue" : Si la quinte est 2x plus forte que la tonique, 
                # l'oreille humaine est souvent perturbée, on réduit le score.
                if quinte_pres > (tonique_pres * 1.8):
                    stability *= 0.75

                final_score = corr * stability
                
                if final_score > best_score: 
                    best_score, best_key = final_score, f"{NOTES_LIST[i]} {mode}"
                    
    return {"key": best_key, "score": min(best_score / 2.5, 1.0)}

def process_audio(file_buffer, file_name, progress_bar, status_text):
    try:
        status_text.text(f"Analyse perceptive : {file_name}")
        y, sr = librosa.load(file_buffer, sr=22050)
        
        # 1. PSYCHO-FILTRAGE : On retire ce que l'oreille ignore (bruits extrêmes)
        # On utilise une pré-emphase pour simuler la courbe de réponse de l'oreille
        y = librosa.effects.preemphasis(y)

        # 2. HPSS AGRESSIF : On isole les instruments mélodiques (le 'chant' et les accords)
        status_text.text("Séparation des composantes harmoniques...")
        y_harmonic = librosa.effects.harmonic(y, margin=10.0) 
        
        # 3. TUNING PRÉCIS
        tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
        
        # 4. CQT AVEC FILTRE DE BASSES (C'est ici qu'on imite l'oreille)
        # On descend à C1 (32Hz) pour bien 'entendre' la tonique des basses
        status_text.text("Extraction de l'empreinte tonale...")
        chroma = librosa.feature.chroma_cqt(
            y=y_harmonic, sr=sr, tuning=tuning, 
            n_chroma=12, bins_per_octave=24, fmin=librosa.note_to_hz('C1')
        )
        
        # 5. LISSAGE COGNITIF (L'oreille moyenne les notes sur la durée)
        chroma_smooth = scipy.ndimage.median_filter(chroma, size=(1, 61)) # Fenêtre plus large
        full_chroma_avg = np.mean(chroma_smooth, axis=1)
        
        # 6. DÉTECTION ET BPM
        res_logic = solve_key_logic(full_chroma_avg)
        status_text.text("Calcul du rythme cardiaque (BPM)...")
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
    for f in uploaded_files:
        pbar = st.progress(0)
        stext = st.empty()
        res = process_audio(f, f.name, pbar, stext)
        pbar.empty()
        stext.empty()

        if "error" in res:
            st.error(f"Erreur : {res['error']}")
            continue

        with st.expander(f"🎵 ANALYSE : {res['name']}", expanded=True):
            c1, c2 = st.columns([1, 1.2])
            with c1:
                color = "#10b981" if res['fiabilite'] > 65 else "#6366f1"
                st.markdown(f"""
                    <div style="background:#111827; padding:20px; border-radius:12px; border:2px solid {color};">
                        <h4 style="margin:0; opacity:0.6;">CLÉ PERCEPTIVE</h4>
                        <h1 style="font-size:50px; color:{color}; margin:10px 0;">{res['key'].upper()}</h1>
                        <p style="font-size:20px;"><b>CAMELOT : {res['camelot']}</b> | <b>{res['tempo']} BPM</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                st.info(f"**Confiance Auditive : {res['fiabilite']}%**")
                
                # Bouton de test tonique
                st.write("🎹 **Vérifier la tonique :**")
                from generate_reference_chord import generate_reference_chord # Hypothèse que la fonction est accessible
                chord = generate_reference_chord(res['key'], sr=res['sr'])
                st.audio(chord, sample_rate=res['sr'])

            with c2:
                # Radar Graph
                r_vals = np.append(res['chroma_vals'], res['chroma_vals'][0])
                theta_vals = NOTES_LIST + [NOTES_LIST[0]]
                fig = go.Figure(go.Scatterpolar(r=r_vals, theta=theta_vals, fill='toself', line_color=color))
                fig.update_layout(template="plotly_dark", height=350, margin=dict(l=20,r=20,b=20,t=20), polar=dict(radialaxis=dict(visible=False)))
                st.plotly_chart(fig, use_container_width=True)

st.caption("Engine v3.7 | Mode Perception Humaine Active")
