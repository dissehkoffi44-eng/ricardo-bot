import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import requests
import os
import tempfile
import streamlit.components.v1 as components

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Psycho-Engine", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES ---
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

# --- FONCTIONS DE COMMUNICATION ---
def send_telegram_message(message):
    """Envoie les résultats sur Telegram (Code 1)."""
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try:
            requests.post(url, json=payload, timeout=5)
        except:
            pass

# --- FONCTIONS TECHNIQUES ---
def apply_perceptual_filter(y, sr):
    """Simule l'oreille humaine via pondération A."""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    a_weights = librosa.perceptual_weighting(S**2, freqs)
    return librosa.istft(S * librosa.db_to_amplitude(a_weights))

def solve_key_logic(chroma_vector):
    """Moteur de détection utilisant les profils de corrélation."""
    best_score, best_key, winners = -1, "", {}
    # Normalisation
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > p_max: 
                    p_max, p_note = score, f"{NOTES_LIST[i]} {mode}"
                
                # Pondération légère pour Krumhansl (plus robuste)
                t_score = score * 1.1 if p_name == "krumhansl" else score
                if t_score > best_score: 
                    best_score, best_key = t_score, f"{NOTES_LIST[i]} {mode}"
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "details": winners}

def play_chord_button(note_mode, uid):
    """Générateur de son JS pour vérifier la tonalité."""
    if " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s").replace(" ","")
    return components.html(f"""
    <button id="{js_id}" style="background:linear-gradient(90deg, #6366F1, #8B5CF6); color:white; border:none; border-radius:12px; padding:15px; cursor:pointer; width:100%; font-weight:bold;">
        🔊 TESTER {n} {m.upper()}
    </button>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    document.getElementById('{js_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)(); const now = ctx.currentTime;
        const intervals = '{m}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(it => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, it/12), now);
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.2, now+0.1); g.gain.exponentialRampToValueAtTime(0.01, now+1.5);
            o.connect(g); g.connect(ctx.destination); o.start(now); o.stop(now+1.5);
        }});
    }};
    </script>""", height=110)

def process_audio(file_buffer, file_name, progress_bar, status_text):
    """Traitement complet de l'audio par segmentation."""
    try:
        duration = librosa.get_duration(path=file_buffer)
        sr = 22050
        
        # Estimation Tuning
        y_tuning, _ = librosa.load(file_buffer, sr=sr, offset=max(0, duration/2 - 10), duration=20)
        tuning = librosa.estimate_tuning(y=y_tuning, sr=sr)
        
        step, timeline, votes = 8, [], Counter()
        segments = list(range(0, int(duration) - step, step // 2))
        all_chromas = []

        for idx, start in enumerate(segments):
            progress_bar.progress((idx + 1) / len(segments))
            status_text.text(f"Analyse Psycho-Acoustique : {int((idx+1)/len(segments)*100)}%")
            
            y_seg, _ = librosa.load(file_buffer, sr=sr, offset=start, duration=step)
            if np.max(np.abs(y_seg)) < 0.01: continue 

            y_seg = apply_perceptual_filter(y_seg, sr)
            # Utilisation de CQT pour la précision harmonique (Code 1)
            chroma = librosa.feature.chroma_cqt(y=y_seg, sr=sr, tuning=tuning, n_chroma=12)
            mean_chroma_seg = np.mean(chroma, axis=1)
            all_chromas.append(mean_chroma_seg)
            
            res = solve_key_logic(mean_chroma_seg)
            votes[res['key']] += (res['score'] ** 2)
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        final_key = votes.most_common(1)[0][0]
        full_chroma_avg = np.mean(all_chromas, axis=0)
        
        # Tempo (version légère)
        file_buffer.seek(0)
        y_light, sr_light = librosa.load(file_buffer, sr=11025, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y_light, sr=sr_light)

        camelot_key = final_key.split(' ')[0]
        camelot_code = (BASE_CAMELOT_MINOR if 'minor' in final_key else BASE_CAMELOT_MAJOR).get(camelot_key, "??")

        return {
            "name": file_name, "tempo": int(float(tempo)), "key": final_key,
            "camelot": camelot_code,
            "conf": int(pd.DataFrame(timeline)['Conf'].mean()) if timeline else 0,
            "details": solve_key_logic(full_chroma_avg)['details'],
            "timeline": timeline,
            "chroma_vals": full_chroma_avg
        }
    except Exception as e: return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---
st.title("🎧 RCDJ228 M1 PRO - Psycho-Engine")
st.markdown("---")

uploaded_files = st.file_uploader("📂 Glissez vos fichiers audio", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in reversed(uploaded_files):
        st.divider()
        pbar = st.progress(0); stext = st.empty()
        
        res = process_audio(f, f.name, pbar, stext)
        pbar.empty(); stext.empty()

        if "error" in res:
            st.error(res['error']); continue

        # --- AFFICHAGE DES RÉSULTATS ---
        with st.expander(f"💎 RÉSULTATS : {res['name']}", expanded=True):
            
            col_main_1, col_main_2 = st.columns([1, 1])
            
            with col_main_1:
                st.markdown(f'''
                    <div style="background:linear-gradient(135deg, #1e3a8a, #581c87); padding:30px; border-radius:20px; text-align:center; color:white; border: 1px solid #6366f1;">
                        <h3 style="margin:0; font-size:1.2rem; opacity:0.8;">TONALITÉ FINALE</h3>
                        <h1 style="margin:10px 0; font-size:3.5rem; color:white;">{res['key'].upper()}</h1>
                        <p style="font-size:1.5rem; font-weight:bold; margin:0;">CAMELOT: {res['camelot']} | {res['conf']}% Fiabilité</p>
                    </div>
                ''', unsafe_allow_html=True)

            with col_main_2:
                # Graphique Radar de l'énergie des notes (Moteur Code 1)
                categories = NOTES_LIST
                fig_radar = go.Figure(data=go.Scatterpolar(
                    r=res['chroma_vals'], theta=categories, fill='toself', line_color='#00FFAA'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=False)), height=250,
                    margin=dict(l=20, r=20, t=40, b=20), template="plotly_dark", title="Empreinte Harmonique"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Métriques secondaires
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Tempo Estimé", f"{res['tempo']} BPM")
            with c2: play_chord_button(res['key'], f"btn_{f.name}")
            with c3: 
                models_str = " | ".join([f"{k}: {v}" for k,v in res['details'].items()])
                st.caption(f"Analyse multi-modèles : {models_str}")

            # Timeline de stabilité (Code 2)
            st.plotly_chart(px.line(
                pd.DataFrame(res['timeline']), x="Temps", y="Note", 
                title="Stabilité de la tonalité sur la durée",
                markers=True, category_orders={"Note": NOTES_ORDER}, template="plotly_dark"
            ), use_container_width=True)

            # Envoi automatique Telegram (Moteur Code 1)
            msg = f"🎵 *Analyse DJ Ricardo*\n*Fichier:* {res['name']}\n*Résultat:* {res['key']}\n*Camelot:* {res['camelot']}\n*Tempo:* {res['tempo']} BPM"
            send_telegram_message(msg)

else:
    st.info("👋 En attente de fichiers audio pour lancer l'analyse psycho-acoustique.")
