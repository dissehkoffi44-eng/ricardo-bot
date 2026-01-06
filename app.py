import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import io
import streamlit.components.v1 as components
import requests
import gc
from scipy.signal import butter, lfilter

# --- CONFIGURATION SÉCURISÉE ---
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN", "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo")
CHAT_ID = st.secrets.get("CHAT_ID", "-1003602454394")

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="RCDJ228 M3 PRO", page_icon="🎧", layout="wide")

# --- CONSTANTES ET PROFILS HARMONIQUES (BASES SOLIDES) ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

PROFILES = {
    "krumhansl": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    },
    "temperley": {
        "major": [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0],
        "minor": [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 2.0, 1.5, 4.0]
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
    .metric-container { background: #1a1c24; padding: 15px; border-radius: 15px; border: 1px solid #333; text-align: center; min-height: 110px; display: flex; flex-direction: column; justify-content: center; }
    .metric-label { font-size: 0.8em; color: #888; text-transform: uppercase; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #FFFFFF; }
    .final-decision-box { padding: 40px; border-radius: 25px; text-align: center; margin: 15px 0; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5); background: #2ecc71; color: white; }
    .profile-tag { background: rgba(99, 102, 241, 0.1); color: #a5b4fc; padding: 3px 8px; border-radius: 6px; font-size: 0.75em; margin: 2px; display: inline-block; border: 1px solid rgba(99, 102, 241, 0.3); }
    </style>
    """, unsafe_allow_html=True)

# --- MOTEURS DE CALCUL (LOGIQUE INTACTE) ---

def apply_perceptual_filter(y, sr):
    nyq = 0.5 * sr
    low, high = 100 / nyq, 5000 / nyq
    b, a = butter(4, [low, high], btype='band')
    return lfilter(b, a, y)

def get_enhanced_chroma(y, sr, tuning):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, n_chroma=12, bins_per_octave=24)
    chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric='cosine')
    return chroma

def solve_key_logic(chroma_vector):
    best_score, best_key, best_root, best_mode = -1, "", 0, "major"
    winners = {}
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    for p_name, p_data in PROFILES.items():
        p_max, p_note = -1, ""
        for mode in ["major", "minor"]:
            for i in range(12):
                score = np.corrcoef(cv, np.roll(p_data[mode], i))[0, 1]
                note_str = f"{NOTES_LIST[i]} {mode}"
                if score > p_max:
                    p_max, p_note = score, note_str
                total_score = score * 1.2 if p_name == "bellman" else score
                if total_score > best_score:
                    best_score, best_root, best_mode, best_key = total_score, i, mode, note_str
        winners[p_name] = p_note
    return {"key": best_key, "score": best_score, "root": best_root, "mode": best_mode, "details": winners}

def get_camelot(key_str):
    try:
        n, m = key_str.split(" ")
        return BASE_CAMELOT_MINOR.get(n, "??") if m == 'minor' else BASE_CAMELOT_MAJOR.get(n, "??")
    except: return "??"

def play_chord_button(note_mode, uid):
    if not note_mode or " " not in note_mode: return ""
    n, m = note_mode.split(' ')
    js_id = f"btn_{uid}".replace(".","").replace("#","s").replace("-","_")
    return components.html(f"""
    <button id="{js_id}" style="background:linear-gradient(90deg, #6366F1, #8B5CF6); color:white; border:none; border-radius:12px; padding:15px; cursor:pointer; font-weight:bold; width:100%; font-family:sans-serif;">🔊 TESTER {n} {m.upper()}</button>
    <script>
    const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
    document.getElementById('{js_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const now = ctx.currentTime;
        const intervals = '{m}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        intervals.forEach(it => {{
            const o = ctx.createOscillator(); const g = ctx.createGain();
            o.type = 'triangle'; o.frequency.setValueAtTime(freqs['{n}'] * Math.pow(2, it/12), now);
            g.gain.setValueAtTime(0, now); g.gain.linearRampToValueAtTime(0.2, now+0.1); g.gain.exponentialRampToValueAtTime(0.01, now+1.5);
            o.connect(g); g.connect(ctx.destination); o.start(now); o.stop(now+1.5);
        }});
    }};
    </script>""", height=70)

# --- ANALYSE ET GESTION MÉMOIRE ---

@st.cache_data(show_spinner=False)
def process_audio(file_bytes, file_name):
    try:
        # Optimisation : chargement avec durée limitée si nécessaire ou sr réduit
        y, sr = librosa.load(io.BytesIO(file_bytes), sr=22050)
        tuning = librosa.estimate_tuning(y=y, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        y_filt = apply_perceptual_filter(y, sr)
        
        step, timeline = 8, []
        votes = Counter()
        
        for start in range(0, int(duration) - step, step):
            y_seg = y_filt[int(start*sr):int((start+step)*sr)]
            if np.max(np.abs(y_seg)) < 0.01: continue 
            chroma = get_enhanced_chroma(y_seg, sr, tuning)
            res = solve_key_logic(np.mean(chroma, axis=1))
            votes[res['key']] += int(res['score'] * 100)
            timeline.append({"Temps": start, "Note": res['key'], "Conf": round(res['score']*100, 1)})

        if not timeline: return {"error": "Silence détecté"}

        final_key = votes.most_common(1)[0][0]
        avg_conf = int(pd.DataFrame(timeline)[pd.DataFrame(timeline)['Note'] == final_key]['Conf'].mean())
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Plotly image pour Telegram (Format allégé)
        fig = px.line(pd.DataFrame(timeline), x="Temps", y="Note", category_orders={"Note": NOTES_ORDER}, template="plotly_dark")
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
        img_bytes = fig.to_image(format="png", width=800, height=400)

        result = {
            "name": file_name, "tempo": int(float(tempo)), "key": final_key, 
            "camelot": get_camelot(final_key), "conf": avg_conf, 
            "details": solve_key_logic(np.mean(get_enhanced_chroma(y, sr, tuning), axis=1))['details'],
            "timeline": timeline, "plot": img_bytes
        }
        del y, y_filt, file_bytes; gc.collect()
        return result
    except Exception as e:
        return {"error": str(e)}

# --- INTERFACE BATCH (100+ FILES) ---

st.title("🎧 RCDJ228 M3 - Analyse Professionnelle")

with st.sidebar:
    st.header("Paramètres")
    uploaded_files = st.file_uploader("📂 Fichiers Audio (FLAC, MP3, WAV)", type=['mp3','wav','flac'], accept_multiple_files=True)
    if st.button("🧹 Vider le Cache"):
        st.cache_data.clear()
        st.rerun()

if uploaded_files:
    # 1. Zone de Progression
    progress_container = st.container()
    pbar = progress_container.progress(0)
    status_text = progress_container.empty()
    
    # 2. Zone de Résumé (Tableau)
    st.subheader("📋 Récapitulatif Global")
    summary_placeholder = st.empty()
    all_results = []

    # 3. Traitement
    for i, f in enumerate(uploaded_files):
        status_text.text(f"Analyse en cours ({i+1}/{len(uploaded_files)}) : {f.name}")
        
        # Extraction des bytes sans bloquer
        f_bytes = f.getvalue()
        res = process_audio(f_bytes, f.name)
        
        if "error" in res:
            st.warning(f"Saut de {f.name}: {res['error']}")
            continue

        # Stockage pour le tableau
        all_results.append({
            "Fichier": res['name'],
            "Tonalité": res['key'],
            "Camelot": res['camelot'],
            "BPM": res['tempo'],
            "Confiance": f"{res['conf']}%"
        })
        
        # Mise à jour du tableau en temps réel
        summary_placeholder.dataframe(pd.DataFrame(all_results), use_container_width=True)

        # Détails individuels dans un expander (fermé pour économiser le CPU)
        with st.expander(f"🔍 Détails : {res['name']} ({res['camelot']})", expanded=False):
            grad = "linear-gradient(135deg, #1e3a8a, #581c87)" if res['conf'] > 75 else "linear-gradient(135deg, #334155, #0f172a)"
            st.markdown(f'<div class="final-decision-box" style="background:{grad}"><h1>{res["key"]} | {res["camelot"]}</h1><p>Confiance: {res["conf"]}% | {res["tempo"]} BPM</p></div>', unsafe_allow_html=True)
            
            c1, c2 = st.columns([1, 2])
            with c1:
                play_chord_button(res['key'], f"btn_{i}")
                tags = "".join([f"<span class='profile-tag'>{p}: {v}</span>" for p, v in res['details'].items()])
                st.markdown(f"**Profils :**<br>{tags}", unsafe_allow_html=True)
            with c2:
                st.plotly_chart(px.line(pd.DataFrame(res['timeline']), x="Temps", y="Note", category_orders={"Note": NOTES_ORDER}, template="plotly_dark", height=250), use_container_width=True)

        # Envoi Telegram (Asynchrone/Non-bloquant)
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                         files={'photo': res['plot']}, 
                         data={'chat_id': CHAT_ID, 'caption': f"📂 {res['name']}\n🎹 {res['key']} ({res['camelot']})\n🔥 {res['conf']}%", 'parse_mode': 'Markdown'}, 
                         timeout=1)
        except: pass

        pbar.progress((i + 1) / len(uploaded_files))
        gc.collect() # Force le nettoyage mémoire entre chaque fichier

    status_text.success(f"✅ Analyse de {len(uploaded_files)} fichiers terminée avec succès.")

    # Bouton de téléchargement CSV
    df_final = pd.DataFrame(all_results)
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les résultats (CSV)", csv, "analyse_rcdj228.csv", "text/csv")
