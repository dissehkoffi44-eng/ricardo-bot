import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests
from scipy.signal import butter, lfilter

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M1 PRO - Human-Ear Engine", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

# Profils de corrélation psychoacoustique (Krumhansl est le plus proche de la perception humaine)
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

# --- STYLES CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .metric-container { 
        background: #1a1c24; padding: 15px; border-radius: 15px; border: 1px solid #333; 
        text-align: center; min-height: 110px; display: flex; flex-direction: column; justify-content: center;
    }
    .value-custom { font-size: 1.6em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { 
        padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .profile-tag { 
        background: rgba(99, 102, 241, 0.1); color: #a5b4fc; padding: 3px 10px; 
        border-radius: 6px; font-size: 0.75em; margin: 2px; display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS TECHNIQUES ---

def apply_perceptual_filter(y, sr):
    """Simule la courbe de réponse de l'oreille (Pondération A)"""
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    a_weights = librosa.perceptual_weighting(S**2, freqs)
    return librosa.istft(S * librosa.db_to_amplitude(a_weights))

def get_humanized_chroma(y, sr, tuning):
    """
    Combinaison de la hiérarchie des hauteurs et de la perception de la basse.
    Implémente les points 1 à 4 demandés.
    """
    # 1. Extraction harmonique (Cœur de la tonalité)
    y_harm = librosa.effects.harmonic(y, margin=6.0)
    
    # 2. Chroma CQT (Octaves centrales 150-800Hz prioritaires pour l'oreille)
    chroma_map = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, n_chroma=12)
    
    # 3. Focus sur la Tonique (Analyse spécifique des basses fréquences)
    # L'oreille identifie la note de basse comme le centre de gravité
    low_idx = librosa.feature.chroma_cens(y=y_harm, sr=sr, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C3'))
    
    # Fusion Cognitive : 70% Harmonie globale + 30% Fondamentale (Basse)
    combined = (chroma_map * 0.7) + (low_idx * 0.3)
    
    # 4. Lissage temporel (Simule la mémoire échoïque)
    # On évite les changements de notes trop brefs que l'oreille ignore
    chroma_smooth = librosa.decompose.nn_filter(combined, aggregate=np.median, metric='cosine')
    
    # Accentuation du contraste (Non-linéarité de la perception)
    return np.power(chroma_smooth, 3.0)

def solve_key_logic(chroma_vector):
    best_score, best_key, winners = -1, "", {}
    # Normalisation Min-Max
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                if score > p_max: 
                    p_max, p_note = score, f"{NOTES_LIST[i]} {mode}"
                
                # Boost pour le profil Krumhansl (le plus humain)
                t_score = score * 1.1 if p_name == "krumhansl" else score
                if t_score > best_score: 
                    best_score, best_key = t_score, f"{NOTES_LIST[i]} {mode}"
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "details": winners}

def get_consonance_score(chroma_vec, key_str):
    try:
        note, mode = key_str.split(" ")
        idx = NOTES_LIST.index(note)
        target = np.zeros(12)
        intervals = [0, 4, 7] if mode == "major" else [0, 3, 7]
        for i in intervals: target[(idx + i) % 12] = 1.0
        score = np.dot(chroma_vec, target) / (np.linalg.norm(chroma_vec) * np.linalg.norm(target) + 1e-6)
        return int(score * 100)
    except: return 0

def play_chord_button(note_mode, uid):
    if " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s")
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

# --- CŒUR DE L'ANALYSE ---

def process_audio(file_bytes, file_name, progress_bar, status_text):
    try:
        # Chargement et Normalisation
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
        y = librosa.util.normalize(y)
        
        # Désaccordage (Tuning)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        
        # Filtre Perceptuel de l'oreille
        y_filt = apply_perceptual_filter(y, sr)
        
        step, timeline, votes = 6, [], Counter()
        duration = int(librosa.get_duration(y=y))
        segments = list(range(0, duration - step, step // 2)) # Overlap pour plus de fluidité cognitive

        for idx, start in enumerate(segments):
            progress_bar.progress((idx + 1) / len(segments))
            status_text.text(f"Psych-Analysis : {int((idx+1)/len(segments)*100)}%")
            
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            if np.max(np.abs(y_seg)) < 0.02: continue 

            # Analyse avec le moteur "Humanized"
            chroma_human = get_humanized_chroma(y_seg, sr, tuning)
            res = solve_key_logic(np.mean(chroma_human, axis=1))
            
            # Vote pondéré par la confiance (score au cube pour favoriser l'évidence)
            votes[res['key']] += (res['score'] ** 3)
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        # Résultats finaux
        final_key = votes.most_common(1)[0][0]
        full_chroma = np.mean(get_humanized_chroma(y_filt, sr, tuning), axis=1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        return {
            "name": file_name, "tempo": int(float(tempo)), "key": final_key,
            "camelot": (BASE_CAMELOT_MINOR if 'minor' in final_key else BASE_CAMELOT_MAJOR).get(final_key.split(' ')[0], "??"),
            "conf": int(pd.DataFrame(timeline)['Conf'].mean()),
            "consonance": get_consonance_score(full_chroma, final_key),
            "details": solve_key_logic(full_chroma)['details'],
            "timeline": timeline
        }
    except Exception as e: return {"error": str(e)}

# --- INTERFACE UTILISATEUR ---

st.title("🎧 RCDJ228 M1 PRO - Psycho-Acoustic Engine")
st.caption("Détection de tonalité basée sur la perception humaine (Hiérarchie des hauteurs & Pondération des basses)")

uploaded_files = st.file_uploader("📂 Charger des pistes (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

if uploaded_files:
    for f in reversed(uploaded_files):
        st.divider()
        pbar = st.progress(0); stext = st.empty()
        res = process_audio(f.read(), f.name, pbar, stext)
        pbar.empty(); stext.empty()

        if "error" in res:
            st.error(f"Erreur sur {f.name}: {res['error']}"); continue

        with st.expander(f"💎 ANALYSE TERMINÉE : {res['name']}", expanded=True):
            # Sélecteur d'ajustement (permet à l'utilisateur de comparer les profils)
            potential_keys = list(set([res['key']] + list(res['details'].values())))
            sel_key = st.selectbox(f"Clé détectée (Ajustement possible)", potential_keys, 
                                   index=potential_keys.index(res['key']), 
                                   key=f"sel_{f.name}")
            
            is_minor = 'minor' in sel_key
            cur_cam = (BASE_CAMELOT_MINOR if is_minor else BASE_CAMELOT_MAJOR).get(sel_key.split(' ')[0], "??")
            
            # Affichage Principal
            bg_gradient = "linear-gradient(135deg, #0f172a, #581c87)" if is_minor else "linear-gradient(135deg, #1e3a8a, #0369a1)"
            st.markdown(f'''
                <div class="final-decision-box" style="background:{bg_gradient}; color: white;">
                    <h1 style="color: white; margin:0;">{sel_key.upper()}</h1>
                    <p style="color: rgba(255,255,255,0.8); font-size:1.2em;">CAMELOT {cur_cam} | Stabilité: {res["conf"]}%</p>
                </div>
            ''', unsafe_allow_html=True)

            # Métriques
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.markdown(f'<div class="metric-container">Tempo<br><span class="value-custom">{res["tempo"]} BPM</span></div>', unsafe_allow_html=True)
            with c2: play_chord_button(sel_key, f"btn_{f.name}")
            with c3: st.markdown(f'<div class="metric-container">Pureté Tonale<br><span class="value-custom">{res["consonance"]}%</span></div>', unsafe_allow_html=True)
            with c4: 
                tags = "".join([f"<span class='profile-tag'>{v}</span>" for v in res['details'].values()])
                st.markdown(f'<div class="metric-container">Modèles<br>{tags}</div>', unsafe_allow_html=True)

            # Graphique de stabilité temporelle (Point 4 : Perception du mouvement)
            df_tl = pd.DataFrame(res['timeline'])
            fig = px.line(df_tl, x="Temps", y="Note", markers=True, 
                         title="Évolution de l'ancrage tonal (Mémoire courte)",
                         category_orders={"Note": NOTES_ORDER}, template="plotly_dark")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

            # Rapport Telegram
            if st.button(f"🚀 Envoyer Rapport : {res['name']}", key=f"tele_{f.name}"):
                if TELEGRAM_TOKEN and CHAT_ID:
                    try:
                        msg = f"🎧 *REPORT PSYCHO-ENGINE*\n📁 `{res['name']}`\n🎹 `{sel_key}` ({cur_cam})\n⏱ `{res['tempo']} BPM`"
                        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                                      data={'chat_id': CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'})
                        st.success("Rapport envoyé !")
                    except: st.error("Erreur d'envoi Telegram")
                else: st.warning("Secrets Telegram non configurés.")

if st.sidebar.button("Réinitialiser le moteur"):
    st.cache_data.clear()
    st.rerun()
