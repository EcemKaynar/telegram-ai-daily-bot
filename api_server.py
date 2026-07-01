import base64
import os
import tempfile
from typing import Any, List, Optional

import telebot
from telebot import types
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from database import init_db
from core_service import (
    process_text_message,
    process_voice_message,
    process_image_message
)

from admin_service import (
    get_admin_overview,
    get_sessions,
    get_session_detail,
    get_service_logs,
    get_rag_logs,
    get_evaluation_results,
    get_evaluation_summary,
    get_admin_data
)


load_dotenv(".env")

init_db()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "telegram-ai-bot-secret")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

bot = None

if TELEGRAM_BOT_TOKEN:
    bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


app = FastAPI(
    title="Telegram AI Daily Bot API",
    description="Telegram'dan bağımsız çalışan günlük yaşam asistanı backend API servisi.",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if os.path.exists("web"):
    app.mount("/web", StaticFiles(directory="web", html=True), name="web")


class ChatRequest(BaseModel):
    user_id: int
    username: str = "web_user"
    first_name: str = ""
    message: str


class ChatResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    input_text: Optional[str] = None
    transcript: Optional[str] = None
    caption: Optional[str] = None
    answer: str
    source_type: Optional[str] = None
    sources: List[Any] = []
    audio_base64: Optional[str] = None


def get_user_info_from_telegram_message(message):
    user_id = message.from_user.id

    telegram_username = message.from_user.username
    first_name = message.from_user.first_name or ""
    last_name = message.from_user.last_name or ""

    display_name = f"{first_name} {last_name}".strip()
    username = telegram_username or display_name or "unknown_user"

    return user_id, username, first_name


@app.get("/")
def root():
    return {
        "message": "Telegram AI Daily Bot API is running.",
        "web": "/web",
        "admin": "/admin",
        "docs": "/docs",
        "telegram_webhook": f"/telegram/{WEBHOOK_SECRET}"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = process_text_message(
        user_id=request.user_id,
        username=request.username,
        first_name=request.first_name,
        user_text=request.message
    )

    return ChatResponse(
        success=result["success"],
        session_id=result.get("session_id"),
        input_text=result.get("input_text"),
        answer=result["answer"],
        source_type=result.get("source_type"),
        sources=result.get("sources", [])
    )


@app.post("/voice", response_model=ChatResponse)
async def voice(
    user_id: int = Form(...),
    username: str = Form("web_user"),
    first_name: str = Form(""),
    language: str = Form("tr-TR"),
    file: UploadFile = File(...)
):
    temp_audio_path = None
    tts_voice_path = None

    try:
        file_extension = os.path.splitext(file.filename or "")[1]

        if not file_extension:
            file_extension = ".webm"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_audio_path = temp_file.name

        result = process_voice_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            audio_path=temp_audio_path,
            language=language
        )

        audio_base64 = None
        tts_voice_path = result.get("voice_path")

        if tts_voice_path and os.path.exists(tts_voice_path):
            with open(tts_voice_path, "rb") as voice_file:
                audio_base64 = base64.b64encode(voice_file.read()).decode("utf-8")

        return ChatResponse(
            success=result["success"],
            session_id=result.get("session_id"),
            transcript=result.get("transcript"),
            answer=result["answer"],
            source_type=result.get("source_type"),
            sources=result.get("sources", []),
            audio_base64=audio_base64
        )

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

        if tts_voice_path and os.path.exists(tts_voice_path):
            os.remove(tts_voice_path)


@app.post("/image", response_model=ChatResponse)
async def image(
    user_id: int = Form(...),
    username: str = Form("web_user"),
    first_name: str = Form(""),
    caption: str = Form(""),
    file: UploadFile = File(...)
):
    temp_image_path = None

    try:
        file_extension = os.path.splitext(file.filename or "")[1]

        if not file_extension:
            file_extension = ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_image_path = temp_file.name

        result = process_image_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            image_path=temp_image_path,
            caption=caption
        )

        return ChatResponse(
            success=result["success"],
            session_id=result.get("session_id"),
            caption=result.get("caption"),
            answer=result["answer"],
            source_type=result.get("source_type"),
            sources=result.get("sources", [])
        )

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)


@app.get("/admin")
def admin_page():
    admin_file = os.path.join("web", "admin.html")

    if os.path.exists(admin_file):
        return FileResponse(admin_file)

    return {
        "error": "admin.html bulunamadı."
    }


@app.get("/admin-data")
def admin_data():
    return get_admin_data()


@app.get("/admin-data/overview")
def admin_overview():
    return get_admin_overview()


@app.get("/admin-data/sessions")
def admin_sessions():
    return get_sessions()


@app.get("/admin-data/sessions/{session_id}")
def admin_session_detail(session_id: str):
    return get_session_detail(session_id)


@app.get("/admin-data/service-logs")
def admin_service_logs(session_id: Optional[str] = None):
    return get_service_logs(session_id=session_id)


@app.get("/admin-data/rag-logs")
def admin_rag_logs(session_id: Optional[str] = None):
    return get_rag_logs(session_id=session_id)


@app.get("/admin-data/evaluation-results")
def admin_evaluation_results():
    return {
        "summary": get_evaluation_summary(),
        "results": get_evaluation_results()
    }


@app.get("/telegram/set-webhook")
def set_telegram_webhook():
    if not bot:
        return {
            "success": False,
            "message": "TELEGRAM_BOT_TOKEN bulunamadı."
        }

    if not WEBHOOK_URL:
        return {
            "success": False,
            "message": "WEBHOOK_URL bulunamadı. Render canlı linkini environment variable olarak eklemelisin."
        }

    full_webhook_url = WEBHOOK_URL.rstrip("/") + f"/telegram/{WEBHOOK_SECRET}"

    bot.remove_webhook()
    success = bot.set_webhook(url=full_webhook_url)

    return {
        "success": success,
        "webhook_url": full_webhook_url
    }


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request):
    if secret != WEBHOOK_SECRET:
        return {
            "success": False,
            "message": "Invalid webhook secret."
        }

    if not bot:
        return {
            "success": False,
            "message": "Telegram bot token is missing."
        }

    update_json = await request.json()
    update = types.Update.de_json(update_json)

    try:
        if update.message:
            message = update.message
            user_id, username, first_name = get_user_info_from_telegram_message(message)

            if message.text:
                if message.text.startswith("/start"):
                    welcome_text = (
                        "Merhaba! Ben günlük yaşamını kolaylaştırmak için hazırlanmış yapay zeka destekli bir Telegram botuyum. "
                        "Günlük planlama, verimlilik, motivasyon, rutin oluşturma, çalışma düzeni, "
                        "mola yönetimi, kahve/çay gibi günlük rutinler, basit yemek önerileri, "
                        "plan defteri, çalışma masası ve hava durumuna göre günlük hazırlık konularında yardımcı olabilirim. "
                        "Bana yazılı mesaj, fotoğraf veya sesli mesaj gönderebilirsin."
                    )

                    bot.reply_to(message, welcome_text)

                else:
                    bot.send_chat_action(message.chat.id, "typing")

                    result = process_text_message(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        user_text=message.text
                    )

                    bot.reply_to(message, result.get("answer", "Cevap alınamadı."))

            elif message.voice:
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

                    result = process_voice_message(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        audio_path=temp_voice_path,
                        language="tr-TR"
                    )

                    transcript = result.get("transcript", "")
                    answer = result.get("answer", "Cevap alınamadı.")

                    text_reply = (
                        f"Sesli mesajını şöyle anladım:\n"
                        f"{transcript}\n\n"
                        f"Cevabım:\n"
                        f"{answer}"
                    )

                    bot.reply_to(message, text_reply)

                    tts_voice_path = result.get("voice_path")

                    if tts_voice_path and os.path.exists(tts_voice_path):
                        with open(tts_voice_path, "rb") as voice_file:
                            bot.send_voice(
                                chat_id=message.chat.id,
                                voice=voice_file,
                                caption="Sesli cevap",
                                reply_to_message_id=message.message_id
                            )

                finally:
                    if temp_voice_path and os.path.exists(temp_voice_path):
                        os.remove(temp_voice_path)

                    if tts_voice_path and os.path.exists(tts_voice_path):
                        os.remove(tts_voice_path)

            elif message.photo:
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

                    result = process_image_message(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        image_path=temp_image_path,
                        caption=message.caption or ""
                    )

                    bot.reply_to(message, result.get("answer", "Cevap alınamadı."))

                finally:
                    if temp_image_path and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

        return {
            "success": True
        }

    except Exception as error:
        print(f"Telegram webhook error: {error}")

        return {
            "success": False,
            "message": str(error)
        }