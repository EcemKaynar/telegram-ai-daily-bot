import os
import subprocess
import tempfile

import imageio_ffmpeg
import speech_recognition as sr
from gtts import gTTS


def get_ffmpeg_path():
    return imageio_ffmpeg.get_ffmpeg_exe()


def convert_audio_to_wav(audio_path):
    wav_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    ffmpeg_path = get_ffmpeg_path()

    command = [
        ffmpeg_path,
        "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", "16000",
        wav_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Ses wav formatına çevrilemedi: {result.stderr}")

    return wav_path


def speech_to_text(audio_path, language="tr-TR"):
    wav_path = None

    try:
        wav_path = convert_audio_to_wav(audio_path)

        recognizer = sr.Recognizer()

        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_google(
            audio_data,
            language=language
        )

        return {
            "success": True,
            "text": text
        }

    except sr.UnknownValueError:
        return {
            "success": False,
            "message": "Sesli mesajı anlayamadım. Biraz daha net ve kısa şekilde tekrar gönderebilir misin?"
        }

    except sr.RequestError as error:
        return {
            "success": False,
            "message": f"Ses yazıya çevirme servisine ulaşılamadı: {error}"
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Sesli mesaj işlenirken bir hata oluştu: {error}"
        }

    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)


def convert_mp3_to_ogg(mp3_path):
    ogg_path = tempfile.NamedTemporaryFile(delete=False, suffix=".ogg").name
    ffmpeg_path = get_ffmpeg_path()

    command = [
        ffmpeg_path,
        "-y",
        "-i", mp3_path,
        "-c:a", "libopus",
        "-b:a", "32k",
        ogg_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"MP3 ogg formatına çevrilemedi: {result.stderr}")

    return ogg_path


def text_to_speech(text, language="tr"):
    mp3_path = None
    ogg_path = None

    try:
        clean_text = text.strip()

        if len(clean_text) > 1000:
            clean_text = clean_text[:1000]

        mp3_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name

        tts = gTTS(
            text=clean_text,
            lang=language,
            slow=False
        )

        tts.save(mp3_path)

        ogg_path = convert_mp3_to_ogg(mp3_path)

        return {
            "success": True,
            "voice_path": ogg_path
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Cevap sese çevrilirken bir hata oluştu: {error}"
        }

    finally:
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)