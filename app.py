import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import io
import requests
import gc
import streamlit.components.v1 as components
from scipy.signal import butter, lfilter

# --- CONFIGURATION SYSTÈME ---
st.set_page_config(page_title="DJ's Ear Pro Music Elite", page_icon="🎼", layout="wide")

# Récupération des secrets (à configurer dans votre dashboard Streamlit ou secrets.toml)
TELEGRAM_TOKEN = st.secrets.get("TELEGRAM_TOKEN")
CHAT_ID = st.secrets.get("CHAT_ID")

# --- RÉFÉRENTIELS HARMONIQUES ---
NOTES_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
NOTES_ORDER = [f"{n} {m}" for n in NOTES_LIST for m in ['major', 'minor']]

CAMELOT_MAP = {
    'C major': '8B', 'C# major': '3B', 'D major': '10B', 'D# major': '5B', 'E major': '12B', 'F major': '7B',
    'F# major': '2B', 'G major': '9B', 'G# major': '4B', 'A major': '11B', 'A# major': '6B', 'B major': '1B',
    'C minor': '5A', 'C# minor': '12A', 'D minor': '7A', 'D# minor': '2A', 'E minor': '9A', 'F minor': '4A',
    'F# minor': '11A', 'G minor': '6A', 'G# minor': '1A', 'A minor': '8A', 'A# minor': '3A', 'B minor': '10A'
}

PROFILES = {
    "krumhansl_kessler": {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    }
}

# --- STYLES CSS ---
st.markdown("""
    <style>
    .report-card { 
        padding: 40px; border-radius: 25px; text-align: center; color: white; 
        border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .modulation-alert {
        background: rgba(239, 68, 68, 0.2); color: #f87171;
        padding: 15px; border-radius: 12px; border: 1px solid #ef4444;
        margin-top: 20px; font-weight: bold; font-size: 1.1em;
    }
    .metric-box {
        background: #1a1c24; border-radius: 15px; padding: 15px; text-align: center; border: 1px solid #333;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE TRAITEMENT ---

def apply_filters(y, sr):
    y_harm, _ = librosa.effects.hpss(y, margin=(4.0, 1.0))
    y_harm = librosa.effects.preemphasis(y_harm)
    nyq = 0.5 * sr
    b, a = butter(4, [100/nyq, 3000/nyq], btype='band')
    return lfilter(b, a, y_harm)

def solve_key(chroma_vector, global_dom_root=None):
    best_score = -1
    res = {"key": "Inconnu", "score": 0}
    cv = (chroma_vector - chroma_vector.min()) / (chroma_vector.max() - chroma_vector.min() + 1e-6)
    
    for _, p_data in PROFILES.items():
        for mode in ["major", "minor"]:
            for i in range(12):
                rotated_profile = np.roll(p_data[mode], i)
                corr_score = np.corrcoef(cv, rotated_profile)[0, 1]
                
                third_idx = (i + 3) % 12 if mode == "minor" else (i + 4) % 12
                fifth_idx = (i + 7) % 12
                
                dom_bonus = 0
                if global_dom_root is not None:
                    if (i + 7) % 12 == global_dom_root and cv[global_dom_root] > 0.35:
                        dom_bonus = 0.18

                final_score = corr_score + (0.15 * cv[third_idx]) + (0.05 * cv[fifth_idx]) + dom_bonus

                if final_score > best_score:
                    best_score = final_score
                    res = {"key": f"{NOTES_LIST[i]} {mode}", "score": corr_score}
    return res

@st.cache_data(show_spinner=False)
def analyze_full_engine(file_bytes, file_name):
    with io.BytesIO(file_bytes) as b:
        y, sr = librosa.load(b, sr=22050, mono=True)
    
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_filt = apply_filters(y, sr)
    
    chroma_complex = librosa.feature.chroma_cqt(y=y_filt, sr=sr, tuning=tuning, bins_per_octave=24, hop_length=512)
    global_chroma_avg = np.mean(chroma_complex, axis=1)
    
    top_2_idx = np.argsort(global_chroma_avg)[-2:]
    n_p, n_s = top_2_idx[1], top_2_idx[0]
    global_dom_root = n_s if (n_p + 7) % 12 == n_s else (n_p if (n_s + 7) % 12 == n_p else None)

    duration = librosa.get_duration(y=y, sr=sr)
    step = 6 
    timeline = []
    votes = Counter()
    
    for start in range(0, int(duration) - step, step):
        seg = y_filt[int(start*sr):int((start+step)*sr)]
        if np.max(np.abs(seg)) < 0.01: continue
        
        c_seg = librosa.feature.chroma_cqt(y=seg, sr=sr, tuning=tuning, bins_per_octave=24)
        c_avg = np.mean(c_seg, axis=1)
        res = solve_key(c_avg, global_dom_root=global_dom_root)
        
        weight = 1.5 if (start < 15 or start > (duration - 15)) else 1.0
        votes[res['key']] += int(res['score'] * 100 * weight)
        timeline.append({"Temps": start, "Note": res['key'], "Conf": res['score']})

    most_common = votes.most_common(2)
    main_key = most_common[0][0]
    main_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == main_key]) * 100)
    
    target_key, target_conf, modulation_detected = None, 0, False
    if len(most_common) > 1:
        second_key = most_common[1][0]
        count_second = sum(1 for t in timeline if t['Note'] == second_key)
        if count_second > (len(timeline) * 0.18): 
            modulation_detected = True
            target_key = second_key
            target_conf = int(np.mean([t['Conf'] for t in timeline if t['Note'] == second_key]) * 100)

    _, y_perc = librosa.effects.hpss(y)
    tempo, _ = librosa.beat.beat_track(y=y_perc, sr=sr)
    
    output = {
        "key": main_key, "camelot": CAMELOT_MAP.get(main_key, "??"),
        "conf": main_conf,
        "tempo": int(float(tempo)),
        "tuning_hz": round(440 * (2**(tuning/12)), 1),
        "pitch_offset": round(tuning, 2),
        "timeline": timeline, "chroma": global_chroma_avg,
        "modulation": modulation_detected, 
        "target_key": target_key,
        "target_conf": target_conf,
        "target_camelot": CAMELOT_MAP.get(target_key, "??") if target_key else None,
        "name": file_name
    }
    del y, y_filt, y_perc; gc.collect()
    return output

def get_piano_js(button_id, key_name):
    if not key_name or " " not in key_name: return ""
    n, mode = key_name.split()
    return f"""
    document.getElementById('{button_id}').onclick = function() {{
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const freqs = {{'C':261.6,'C#':277.2,'D':293.7,'D#':311.1,'E':329.6,'F':349.2,'F#':370.0,'G':392.0,'G#':415.3,'A':440.0,'A#':466.2,'B':493.9}};
        const chord = '{mode}' === 'minor' ? [0, 3, 7, 12] : [0, 4, 7, 12];
        chord.forEach((interval) => {{
            const baseFreq = freqs['{n}'] * Math.pow(2, interval/12);
            [1, 2].forEach((h) => {{
                const osc = ctx.createOscillator(); const g = ctx.createGain();
                osc.type = h === 1 ? 'triangle' : 'sine';
                osc.frequency.setValueAtTime(baseFreq * h, ctx.currentTime);
                g.gain.setValueAtTime(0, ctx.currentTime);
                g.gain.linearRampToValueAtTime(0.1/h, ctx.currentTime + 0.05);
                g.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 1.5);
                osc.connect(g); g.connect(ctx.destination);
                osc.start(); osc.stop(ctx.currentTime + 1.5);
            }});
        }});
    }};
    """

def send_telegram_expert(data, fig_timeline, fig_radar):
    if not TELEGRAM_TOKEN or not CHAT_ID: return
    mod_text = ""
    if data['modulation']:
        mod_text = (f"⚠️ *MODULATION DÉTECTÉE*\n"
                    f"└ Vers : `{data['target_key']}` ({data['target_camelot']})\n"
                    f"└ Confiance Mod : `{data['target_conf']}%` \n\n")

    msg = (f"🎼 *DJ'S EAR PRO ELITE REPORT*\n"
           f"━━━━━━━━━━━━━━━━━━━━\n"
           f"📂 *Fichier:* `{data['name']}`\n\n"
           f"✅ *TONALITÉ PRINCIPALE*\n"
           f"└ Note : `{data['key'].upper()}`\n"
           f"└ Camelot : `{data['camelot']}`\n"
           f"└ Confiance : `{data['conf']}%`\n\n"
           f"{mod_text}"
           f"📊 *MÉTRIQUES*\n"
           f"└ Tempo : `{data['tempo']} BPM`\n"
           f"└ Tuning : `{data['tuning_hz']} Hz` (`{data['pitch_offset']}`c)\n"
           f"━━━━━━━━━━━━━━━━━━━━")
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                     json={"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"})
        for fig, title in [(fig_timeline, "Flux"), (fig_radar, "Signature")]:
            img_bytes = fig.to_image(format="png", engine="kaleido")
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto", 
                         data={"chat_id": CHAT_ID, "caption": f"📊 {title} - {data['name']}"},
                         files={"photo": img_bytes})
    except Exception as e:
        st.error(f"Erreur Telegram: {e}")

# --- INTERFACE ---
st.title("🎧 DJ's Ear Pro Elite")
st.markdown("### Haute Précision & Intelligence Harmonique")

files = st.file_uploader("📂 Déposer vos morceaux (MP3, WAV, FLAC)", type=['mp3','wav','flac'], accept_multiple_files=True)

if files:
    # Inverser la liste pour traiter le dernier fichier en premier
    files_to_process = list(reversed(files))
    
    for f in files_to_process:
        file_bytes = f.read()
        with st.spinner(f"Analyse haute précision de {f.name}..."):
            data = analyze_full_engine(file_bytes, f.name)
        
        with st.expander(f"📊 ANALYSE TERMINÉE : {data['name']}", expanded=True):
            bg = "linear-gradient(135deg, #0f172a, #1e3a8a)" if not data['modulation'] else "linear-gradient(135deg, #1e1b4b, #7f1d1d)"
            st.markdown(f"""
                <div class="report-card" style="background:{bg};">
                    <p style="opacity:0.7; letter-spacing:2px;">TONALITÉ DÉTECTÉE</p>
                    <h1 style="font-size:5.5em; margin:10px 0;">{data['key'].upper()}</h1>
                    <p style="font-size:1.8em;">CAMELOT : <b>{data['camelot']}</b> | CONFIANCE : <b>{data['conf']}%</b></p>
                    {f"<div class='modulation-alert'>⚠️ MODULATION VERS {data['target_key'].upper()} ({data['target_camelot']}) | CONF: {data['target_conf']}%</div>" if data['modulation'] else ""}
                </div>
            """, unsafe_allow_html=True)

            st.write("---")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"<div class='metric-box'><b>TEMPO</b><br><span style='font-size:1.8em;'>{data['tempo']} BPM</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-box' style='margin-top:10px;'><b>TUNING</b><br><span>{data['tuning_hz']} Hz ({data['pitch_offset']} cents)</span></div>", unsafe_allow_html=True)
            with c2:
                btn_id = f"play_{hash(f.name)}"
                st.markdown(f"<center><b>TESTER L'ACCORD</b></center>", unsafe_allow_html=True)
                components.html(f"""<button id="{btn_id}" style="width:100%; height:110px; background:linear-gradient(90deg, #4F46E5, #7C3AED); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.2em; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🎹 JOUER L'ACCORD</button><script>{get_piano_js(btn_id, data['key'])}</script>""", height=130)
            with c3:
                if data['modulation']:
                    m_id = f"mod_{hash(f.name)}"
                    st.markdown(f"<center><b>TESTER MODULATION</b></center>", unsafe_allow_html=True)
                    components.html(f"""<button id="{m_id}" style="width:100%; height:110px; background:linear-gradient(90deg, #ef4444, #b91c1c); color:white; border:none; border-radius:15px; cursor:pointer; font-weight:bold; font-size:1.2em; box-shadow:0 4px 15px rgba(0,0,0,0.3);">🎹 JOUER MOD ({data['target_camelot']})</button><script>{get_piano_js(m_id, data['target_key'])}</script>""", height=130)
                else:
                    st.markdown("<div style='height:130px; display:flex; align-items:center; justify-content:center; opacity:0.3; border:2px dashed #444; border-radius:15px;'>Stabilité Harmonique</div>", unsafe_allow_html=True)

            gl, gr = st.columns([2, 1])
            with gl:
                df_tl = pd.DataFrame(data['timeline'])
                fig_l = px.line(df_tl, x="Temps", y="Note", markers=True, template="plotly_dark", category_orders={"Note": NOTES_ORDER}, title="Flux Harmonique Dynamique")
                st.plotly_chart(fig_l, use_container_width=True)
            with gr:
                fig_r = go.Figure(data=go.Scatterpolar(r=data['chroma'], theta=NOTES_LIST, fill='toself', line_color='#818cf8'))
                fig_r.update_layout(template="plotly_dark", title="Signature Chromatique", polar=dict(radialaxis=dict(visible=False)))
                st.plotly_chart(fig_r, use_container_width=True)

            # --- ENVOI AUTOMATIQUE TELEGRAM ---
            send_telegram_expert(data, fig_l, fig_r)
            st.toast(f"✅ Rapport Telegram envoyé pour {f.name}")

    if st.sidebar.button("🧹 Vider la mémoire"):
        st.cache_data.clear(); st.rerun()
else:
    st.info("Prêt pour l'analyse profonde (24-bins + Tuning + V-i Logic).")
