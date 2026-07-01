import base64
import os
import tempfile

import requests
import telebot
from telebot import types
from flask import Flask, request
from dotenv import load_dotenv


load_dotenv(".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "telegram-ai-bot-secret")
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN bulunamadı. .env dosyasını kontrol et.")

if not WEBHOOK_URL:
    raise ValueError("WEBHOOK_URL bulunamadı. Ngrok linkini .env dosyasına eklemelisin.")


bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)
app = Flask(__name__)

WEBHOOK_PATH = f"/{WEBHOOK_SECRET}"


def get_user_info_from_message(message):
    user_id = message.from_user.id

    telegram_username = message.from_user.username
    first_name = message.from_user.first_name or ""
    last_name = message.from_user.last_name or ""

    display_name = f"{first_name} {last_name}".strip()
    username = telegram_username or display_name or "unknown_user"

    return user_id, username, first_name


def call_chat_api(user_id, username, first_name, message_text):
    url = f"{API_BASE_URL}/chat"

    payload = {
        "user_id": user_id,
        "username": username,
        "first_name": first_name,
        "message": message_text
    }

    response = requests.post(
        url,
        json=payload,
        timeout=90
    )

    response.raise_for_status()

    return response.json()


def call_voice_api(user_id, username, first_name, audio_path):
    url = f"{API_BASE_URL}/voice"

    data = {
        "user_id": str(user_id),
        "username": username,
        "first_name": first_name,
        "language": "tr-TR"
    }

    with open(audio_path, "rb") as audio_file:
        files = {
            "file": ("voice.ogg", audio_file, "audio/ogg")
        }

        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=120
        )

    response.raise_for_status()

    return response.json()


def call_image_api(user_id, username, first_name, image_path, caption=""):
    url = f"{API_BASE_URL}/image"

    data = {
        "user_id": str(user_id),
        "username": username,
        "first_name": first_name,
        "caption": caption
    }

    with open(image_path, "rb") as image_file:
        files = {
            "file": ("image.jpg", image_file, "image/jpeg")
        }

        response = requests.post(
            url,
            data=data,
            files=files,
            timeout=120
        )

    response.raise_for_status()

    return response.json()


@bot.message_handler(commands=["start"])
def send_welcome(message):
    welcome_text = (
        "Merhaba! Ben günlük yaşamını kolaylaştırmak için hazırlanmış yapay zeka destekli bir Telegram botuyum. "
        "Günlük planlama, verimlilik, motivasyon, rutin oluşturma, çalışma düzeni, "
        "mola yönetimi, kahve/çay gibi günlük rutinler, basit yemek önerileri, "
        "plan defteri, çalışma masası ve hava durumuna göre günlük hazırlık konularında yardımcı olabilirim. "
        "Bana yazılı mesaj, fotoğraf veya sesli mesaj gönderebilirsin."
    )

    bot.reply_to(message, welcome_text)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    user_id, username, first_name = get_user_info_from_message(message)

    bot.send_chat_action(message.chat.id, "typing")

    try:
        api_result = call_chat_api(
            user_id=user_id,
            username=username,
            first_name=first_name,
            message_text=message.text
        )

        answer = api_result.get("answer", "API cevabı alınamadı.")

        bot.reply_to(message, answer)

    except Exception as error:
        print(f"Text API çağrı hatası: {error}")
        bot.reply_to(message, "API'ye bağlanırken bir hata oluştu. Lütfen tekrar dener misin?")


@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    user_id, username, first_name = get_user_info_from_message(message)

    bot.send_chat_action(message.chat.id, "typing")

    temp_voice_path = None
    tts_voice_path = None

    try:
        file_info = bot.get_file(message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        file_extension = os.path.splitext(file_info.file_path)[1]

        if not file_extension:
            file_extension = ".ogg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(downloaded_file)
            temp_voice_path = temp_file.name

        api_result = call_voice_api(
            user_id=user_id,
            username=username,
            first_name=first_name,
            audio_path=temp_voice_path
        )

        transcript = api_result.get("transcript", "")
        answer = api_result.get("answer", "API cevabı alınamadı.")
        audio_base64 = api_result.get("audio_base64")

        text_reply = (
            f"Sesli mesajını şöyle anladım:\n"
            f"{transcript}\n\n"
            f"Cevabım:\n"
            f"{answer}"
        )

        bot.reply_to(message, text_reply)

        if audio_base64:
            audio_bytes = base64.b64decode(audio_base64)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".ogg") as temp_tts_file:
                temp_tts_file.write(audio_bytes)
                tts_voice_path = temp_tts_file.name

            with open(tts_voice_path, "rb") as voice_file:
                bot.send_voice(
                    chat_id=message.chat.id,
                    voice=voice_file,
                    caption="Sesli cevap",
                    reply_to_message_id=message.message_id
                )

    except Exception as error:
        print(f"Voice API çağrı hatası: {error}")
        bot.reply_to(message, "Sesli mesaj API'ye gönderilirken bir hata oluştu. Lütfen tekrar dener misin?")

    finally:
        if temp_voice_path and os.path.exists(temp_voice_path):
            os.remove(temp_voice_path)

        if tts_voice_path and os.path.exists(tts_voice_path):
            os.remove(tts_voice_path)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    user_id, username, first_name = get_user_info_from_message(message)

    bot.send_chat_action(message.chat.id, "typing")

    temp_image_path = None

    try:
        largest_photo = message.photo[-1]
        file_info = bot.get_file(largest_photo.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        file_extension = os.path.splitext(file_info.file_path)[1]

        if not file_extension:
            file_extension = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(downloaded_file)
            temp_image_path = temp_file.name

        api_result = call_image_api(
            user_id=user_id,
            username=username,
            first_name=first_name,
            image_path=temp_image_path,
            caption=message.caption or ""
        )

        answer = api_result.get("answer", "API cevabı alınamadı.")

        bot.reply_to(message, answer)

    except Exception as error:
        print(f"Image API çağrı hatası: {error}")
        bot.reply_to(message, "Fotoğraf API'ye gönderilirken bir hata oluştu. Lütfen tekrar dener misin?")

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.route("/", methods=["GET"])
def index():
    return "Telegram API client is running."


@app.route(WEBHOOK_PATH, methods=["POST"])
def webhook():
    if request.headers.get("content-type") == "application/json":
        json_string = request.get_data().decode("utf-8")
        update = types.Update.de_json(json_string)

        bot.process_new_updates([update])

        return "OK", 200

    return "Unsupported Media Type", 415


if __name__ == "__main__":
    full_webhook_url = WEBHOOK_URL.rstrip("/") + WEBHOOK_PATH

    bot.remove_webhook()
    bot.set_webhook(url=full_webhook_url)

    print("Telegram client başlatıldı.")
    print(f"Webhook URL: {full_webhook_url}")
    print(f"API Base URL: {API_BASE_URL}")

    app.run(host="0.0.0.0", port=5000)