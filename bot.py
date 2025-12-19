import os
import librosa
import numpy as np
from collections import Counter
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

# --- LOGIQUE MUSICALE RICARDO_DJ228 ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key, mode):
    """Convertit la tonalité en code Camelot sans erreur de casse"""
    camelot_map = {
        'g# minor': '1A', 'ab minor': '1A', 'b major': '1B', 'cb major': '1B',
        'd# minor': '2A', 'eb minor': '2A', 'f# major': '2B', 'gb major': '2B',
        'bb minor': '3A', 'a# minor': '3A', 'db major': '3B', 'c# major': '3B',
        'f minor': '4A', 'ab major': '4B', 'c minor': '5A', 'eb major': '5B',
        'g minor': '6A', 'bb major': '6B', 'd minor': '7A', 'f major': '7B',
        'a minor': '8A', 'c major': '8B', 'e minor': '9A', 'g major': '9B',
        'b minor': '10A', 'd major': '10B', 'f# minor': '11A', 'gb minor': '11A', 'a major': '11B',
        'c# minor': '12A', 'db minor': '12A', 'e major': '12B'
    }
    search_term = f"{key} {mode}".strip().lower()
    return camelot_map.get(search_term, "??")

def analyze_audio(file_path):
    """Analyse globale : Key, BPM et Camelot"""
    y, sr = librosa.load(file_path)
    
    # 1. Analyse Tonalité Dominante
    y_harm = librosa.effects.hpss(y)[0]
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score = -1
    final_key, final_mode = "", ""
    for i in range(12):
        for mode in ["major", "minor"]:
            profile = MAJOR_PROFILE if mode == "major" else MINOR_PROFILE
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, final_key, final_mode = score, NOTES[i], mode
    
    # 2. Analyse BPM
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = int(round(float(tempo)))
    
    # 3. Camelot
    camelot = get_camelot(final_key, final_mode)
    
    return final_key, final_mode, bpm, camelot

# --- LOGIQUE DU BOT ---
async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    file_to_download = msg.audio or msg.document or msg.voice
    
    if not file_to_download:
        return

    status_msg = await msg.reply_text("🚀 Analyse en cours pour RICARDO_DJ228...")
    
    # Création d'un nom de fichier unique basé sur l'ID Telegram
    temp_file = f"track_{msg.from_user.id}.mp3"
    
    try:
        bot_file = await file_to_download.get_file()
        await bot_file.download_to_drive(temp_file)
        
        key, mode, bpm, camelot = analyze_audio(temp_file)
        
        # RÉPONSE CORRIGÉE (Pas d'underscores seuls pour éviter l'erreur de parsing)
        response = (
            "✅ *ANALYSE TERMINÉE*\n\n"
            f"🎹 *Tonalité :* `{key} {mode.upper()}`\n"
            f"🌀 *Code Camelot :* `{camelot}`\n"
            f"🥁 *Tempo :* `{bpm} BPM`\n\n"
            "🎧 Analyzed by Ricardo DJ228"
        )
        
        await msg.reply_text(response, parse_mode="Markdown")
        
    except Exception as e:
        await msg.reply_text(f"❌ Erreur : {str(e)}")
    
    finally:
        # Nettoyage
        await status_msg.delete()
        if os.path.exists(temp_file):
            os.remove(temp_file)

# --- START ---
if __name__ == "__main__":
    # REMPLACE PAR TON TOKEN CI-DESSOUS
    TOKEN = "7751365982:AAFLbeRoPsDx5OyIOlsgHcGKpI12hopzCYo"
    
    if TOKEN == "TON_API_TOKEN_ICI":
        print("ERREUR : Oublie pas d'insérer ton Token !")
    else:
        print("--- Bot RICARDO_DJ228 Opérationnel ---")
        app = Application.builder().token(TOKEN).build()
        
        # Gestion des fichiers audio et messages vocaux
        app.add_handler(MessageHandler(filters.AUDIO | filters.Document.Category("audio") | filters.VOICE, handle_audio))
        
        app.run_polling()