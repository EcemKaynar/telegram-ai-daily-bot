import os
import json
import sqlite3
import tempfile
import requests
from typing import Any, Optional, List

from fastapi import FastAPI, Request, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from core_service import (
    process_text_message,
    process_voice_message,
    process_image_message
)


app = FastAPI(
    title="Telegram AI Daily Bot API",
    version="1.0.0",
    description="FastAPI backend for Telegram AI Daily Assistant Bot"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
API_TOKEN = os.getenv("API_TOKEN", "")

DATABASE_PATH = os.getenv("DATABASE_PATH", "bot_messages.db")


PROTECTED_PATH_PREFIXES = (
    "/chat",
    "/voice",
    "/image",
    "/admin-data",
)


def is_protected_path(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in PROTECTED_PATH_PREFIXES)


def get_request_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    api_key = request.headers.get("X-API-Key", "")

    if authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "").strip()

    if api_key:
        return api_key.strip()

    return ""


@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    if not API_AUTH_ENABLED:
        return await call_next(request)

    path = request.url.path

    if not is_protected_path(path):
        return await call_next(request)

    if not API_TOKEN:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "detail": "API authentication is enabled but API_TOKEN is not configured."
            }
        )

    request_token = get_request_token(request)

    if request_token != API_TOKEN:
        return JSONResponse(
            status_code=401,
            content={
                "success": False,
                "detail": "Unauthorized. Valid API token is required."
            }
        )

    return await call_next(request)


class ChatRequest(BaseModel):
    user_id: int = 1
    username: Optional[str] = "web_user"
    first_name: Optional[str] = "Web"
    message: str


class ChatResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    input_text: Optional[str] = None
    transcript: Optional[str] = None
    caption: Optional[str] = None
    answer: str
    voice_path: Optional[str] = None
    source_type: Optional[str] = None
    sources: List[Any] = []


@app.get("/")
def root():
    return {
        "success": True,
        "message": "Telegram AI Daily Bot API is running.",
        "health": "/health",
        "web": "/web",
        "admin": "/admin",
        "docs": "/docs",
        "telegram_webhook": "/telegram/{secret}",
        "api_auth_enabled": API_AUTH_ENABLED
    }


@app.get("/health")
def health():
    return {
        "success": True,
        "status": "ok",
        "api_auth_enabled": API_AUTH_ENABLED
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = process_text_message(
            user_id=request.user_id,
            username=request.username,
            first_name=request.first_name,
            user_text=request.message
        )

        if result is None:
            result = {
                "success": False,
                "session_id": None,
                "input_text": request.message,
                "answer": "Cevap üretirken beklenmeyen bir sorun oluştu. Lütfen tekrar dener misin?",
                "source_type": "internal_none_result",
                "sources": []
            }

        return ChatResponse(
            success=result.get("success", False),
            session_id=result.get("session_id"),
            input_text=result.get("input_text", request.message),
            answer=result.get("answer", "Cevap alınamadı."),
            source_type=result.get("source_type"),
            sources=result.get("sources", [])
        )

    except Exception as error:
        print(f"/chat endpoint hatası: {error}")

        return ChatResponse(
            success=False,
            session_id=None,
            input_text=request.message,
            answer="API tarafında cevap oluştururken bir hata oluştu. Lütfen tekrar dener misin?",
            source_type="chat_endpoint_error",
            sources=[]
        )


@app.post("/voice", response_model=ChatResponse)
def voice(
    audio: UploadFile = File(...),
    user_id: int = Form(1),
    username: str = Form("web_user"),
    first_name: str = Form("Web"),
    language: str = Form("tr-TR")
):
    try:
        suffix = os.path.splitext(audio.filename or "")[1] or ".ogg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio.file.read())
            temp_audio_path = temp_file.name

        result = process_voice_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            audio_path=temp_audio_path,
            language=language
        )

        return ChatResponse(
            success=result.get("success", False),
            session_id=result.get("session_id"),
            transcript=result.get("transcript"),
            answer=result.get("answer", "Cevap alınamadı."),
            voice_path=result.get("voice_path"),
            source_type=result.get("source_type"),
            sources=result.get("sources", [])
        )

    except Exception as error:
        print(f"/voice endpoint hatası: {error}")

        return ChatResponse(
            success=False,
            answer="Sesli mesaj işlenirken bir hata oluştu. Lütfen tekrar dener misin?",
            source_type="voice_endpoint_error",
            sources=[]
        )


@app.post("/image", response_model=ChatResponse)
def image(
    image: UploadFile = File(...),
    user_id: int = Form(1),
    username: str = Form("web_user"),
    first_name: str = Form("Web"),
    caption: str = Form("")
):
    try:
        suffix = os.path.splitext(image.filename or "")[1] or ".jpg"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image.file.read())
            temp_image_path = temp_file.name

        result = process_image_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            image_path=temp_image_path,
            caption=caption
        )

        return ChatResponse(
            success=result.get("success", False),
            session_id=result.get("session_id"),
            caption=result.get("caption"),
            answer=result.get("answer", "Cevap alınamadı."),
            source_type=result.get("source_type"),
            sources=result.get("sources", [])
        )

    except Exception as error:
        print(f"/image endpoint hatası: {error}")

        return ChatResponse(
            success=False,
            answer="Fotoğraf işlenirken bir hata oluştu. Lütfen tekrar dener misin?",
            source_type="image_endpoint_error",
            sources=[]
        )


def send_telegram_message(chat_id, text):
    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN tanımlı değil.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    payload = {
        "chat_id": chat_id,
        "text": text
    }

    try:
        response = requests.post(url, json=payload, timeout=20)
        print("Telegram sendMessage:", response.status_code, response.text[:300])
        return response.ok

    except Exception as error:
        print(f"Telegram mesaj gönderme hatası: {error}")
        return False


def get_telegram_file_path(file_id):
    if not TELEGRAM_BOT_TOKEN:
        return None

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"

    try:
        response = requests.get(url, params={"file_id": file_id}, timeout=20)
        data = response.json()

        if not data.get("ok"):
            print("Telegram getFile başarısız:", data)
            return None

        return data.get("result", {}).get("file_path")

    except Exception as error:
        print(f"Telegram getFile hatası: {error}")
        return None


def download_telegram_file(file_id, suffix=".dat"):
    file_path = get_telegram_file_path(file_id)

    if not file_path:
        return None

    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

    try:
        response = requests.get(download_url, timeout=40)

        if not response.ok:
            print("Telegram file download başarısız:", response.status_code)
            return None

        real_suffix = os.path.splitext(file_path)[1] or suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=real_suffix) as temp_file:
            temp_file.write(response.content)
            return temp_file.name

    except Exception as error:
        print(f"Telegram file download hatası: {error}")
        return None


def process_telegram_update(update):
    try:
        message = update.get("message") or update.get("edited_message")

        if not message:
            return

        chat = message.get("chat", {})
        user = message.get("from", {})

        chat_id = chat.get("id")
        user_id = user.get("id", chat_id)
        username = user.get("username") or ""
        first_name = user.get("first_name") or ""

        if not chat_id:
            return

        if "text" in message:
            user_text = message.get("text", "")

            result = process_text_message(
                user_id=user_id,
                username=username,
                first_name=first_name,
                user_text=user_text
            )

            answer = result.get("answer", "Cevap alınamadı.")
            send_telegram_message(chat_id, answer)
            return

        if "voice" in message:
            voice_data = message.get("voice", {})
            file_id = voice_data.get("file_id")

            if not file_id:
                send_telegram_message(chat_id, "Ses dosyası alınamadı.")
                return

            audio_path = download_telegram_file(file_id, suffix=".ogg")

            if not audio_path:
                send_telegram_message(chat_id, "Ses dosyası indirilemedi.")
                return

            result = process_voice_message(
                user_id=user_id,
                username=username,
                first_name=first_name,
                audio_path=audio_path,
                language="tr-TR"
            )

            answer = result.get("answer", "Cevap alınamadı.")
            send_telegram_message(chat_id, answer)
            return

        if "photo" in message:
            photos = message.get("photo", [])

            if not photos:
                send_telegram_message(chat_id, "Fotoğraf alınamadı.")
                return

            largest_photo = photos[-1]
            file_id = largest_photo.get("file_id")
            caption = message.get("caption", "")

            image_path = download_telegram_file(file_id, suffix=".jpg")

            if not image_path:
                send_telegram_message(chat_id, "Fotoğraf indirilemedi.")
                return

            result = process_image_message(
                user_id=user_id,
                username=username,
                first_name=first_name,
                image_path=image_path,
                caption=caption
            )

            answer = result.get("answer", "Cevap alınamadı.")
            send_telegram_message(chat_id, answer)
            return

        send_telegram_message(
            chat_id,
            "Şu an metin, sesli mesaj ve fotoğraf mesajlarını işleyebilirim."
        )

    except Exception as error:
        print(f"Telegram update işleme hatası: {error}")


@app.post("/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request, background_tasks: BackgroundTasks):
    if not WEBHOOK_SECRET or secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid webhook secret")

    try:
        update = await request.json()
    except Exception:
        update = {}

    background_tasks.add_task(process_telegram_update, update)

    return {
        "ok": True
    }


@app.get("/telegram/set-webhook")
def set_telegram_webhook():
    if not TELEGRAM_BOT_TOKEN:
        return {
            "ok": False,
            "error": "TELEGRAM_BOT_TOKEN is not configured."
        }

    if not WEBHOOK_URL:
        return {
            "ok": False,
            "error": "WEBHOOK_URL is not configured."
        }

    if not WEBHOOK_SECRET:
        return {
            "ok": False,
            "error": "WEBHOOK_SECRET is not configured."
        }

    webhook_endpoint = f"{WEBHOOK_URL.rstrip('/')}/telegram/{WEBHOOK_SECRET}"
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/setWebhook"

    payload = {
        "url": webhook_endpoint,
        "drop_pending_updates": True
    }

    try:
        response = requests.post(url, json=payload, timeout=30)

        return {
            "ok": response.ok,
            "telegram_response": response.json()
        }

    except Exception as error:
        return {
            "ok": False,
            "error": str(error)
        }


@app.get("/telegram/webhook-info")
def telegram_webhook_info():
    if not TELEGRAM_BOT_TOKEN:
        return {
            "ok": False,
            "error": "TELEGRAM_BOT_TOKEN is not configured."
        }

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getWebhookInfo"

    try:
        response = requests.get(url, timeout=30)

        return {
            "ok": response.ok,
            "telegram_response": response.json()
        }

    except Exception as error:
        return {
            "ok": False,
            "error": str(error)
        }


def read_sqlite_table(table_name, limit=100):
    if not os.path.exists(DATABASE_PATH):
        return []

    try:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row

        cursor = connection.cursor()

        cursor.execute(
            f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT ?",
            (limit,)
        )

        rows = [dict(row) for row in cursor.fetchall()]

        connection.close()

        return rows

    except Exception as error:
        print(f"Admin table read error for {table_name}: {error}")
        return []


def get_database_tables():
    if not os.path.exists(DATABASE_PATH):
        return []

    try:
        connection = sqlite3.connect(DATABASE_PATH)
        cursor = connection.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )

        tables = [row[0] for row in cursor.fetchall()]

        connection.close()

        return tables

    except Exception as error:
        print(f"Database table list error: {error}")
        return []


@app.get("/admin-data")
def admin_data():
    tables = get_database_tables()

    data = {
        "success": True,
        "database_path": DATABASE_PATH,
        "tables": {},
        "table_names": tables
    }

    for table in tables:
        data["tables"][table] = read_sqlite_table(table, limit=100)

    return data


@app.get("/web", response_class=HTMLResponse)
@app.get("/web/", response_class=HTMLResponse)
def web_page():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>AI Daily Bot Web Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f4f7;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 850px;
            margin: 30px auto;
            background: white;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
            font-size: 24px;
        }

        .top-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 14px;
        }

        button {
            border: none;
            background: #222;
            color: white;
            padding: 10px 14px;
            border-radius: 9px;
            cursor: pointer;
        }

        button.secondary {
            background: #666;
        }

        button.danger {
            background: #b42318;
        }

        input[type="text"] {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 9px;
            font-size: 15px;
        }

        input[type="file"] {
            margin: 8px 0;
        }

        .chat-box {
            height: 420px;
            overflow-y: auto;
            border: 1px solid #eee;
            padding: 14px;
            border-radius: 12px;
            background: #fafafa;
            margin-bottom: 14px;
        }

        .message {
            padding: 10px 12px;
            margin: 8px 0;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.4;
        }

        .user {
            background: #dbeafe;
            margin-left: 80px;
        }

        .bot {
            background: #ecfdf3;
            margin-right: 80px;
        }

        .error {
            background: #fee4e2;
            color: #7a271a;
        }

        .row {
            display: flex;
            gap: 8px;
            margin-bottom: 10px;
        }

        .panel {
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 12px;
            margin-top: 12px;
            background: #fcfcfc;
        }

        .small {
            font-size: 13px;
            color: #666;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>AI Daily Bot Web Chat</h1>

        <div class="top-bar">
            <button onclick="ensureApiToken()">API Token Gir</button>
            <button class="danger" onclick="resetApiToken()">Token Sıfırla</button>
            <a href="/admin" target="_blank">
                <button class="secondary">Admin Panel</button>
            </a>
        </div>

        <p class="small">
            API_AUTH_ENABLED=true ise chat, voice, image ve admin-data istekleri token ister.
        </p>

        <div id="chatBox" class="chat-box"></div>

        <div class="row">
            <input id="messageInput" type="text" placeholder="Mesaj yaz..." onkeydown="handleEnter(event)">
            <button onclick="sendMessage()">Gönder</button>
        </div>

        <div class="panel">
            <strong>Sesli mesaj test</strong><br>
            <input id="voiceInput" type="file" accept="audio/*">
            <button onclick="sendVoice()">Ses Gönder</button>
        </div>

        <div class="panel">
            <strong>Fotoğraf test</strong><br>
            <input id="imageInput" type="file" accept="image/*">
            <input id="captionInput" type="text" placeholder="Fotoğraf açıklaması / caption">
            <button onclick="sendImage()">Fotoğraf Gönder</button>
        </div>
    </div>

    <script>
        function getApiToken() {
            let token = localStorage.getItem("API_TOKEN");

            if (!token) {
                token = prompt("API token giriniz:");

                if (token) {
                    localStorage.setItem("API_TOKEN", token);
                }
            }

            return token || "";
        }

        function ensureApiToken() {
            const token = getApiToken();

            if (token) {
                alert("Token kaydedildi.");
            }
        }

        function resetApiToken() {
            localStorage.removeItem("API_TOKEN");
            alert("Token sıfırlandı. Yeni istek atarken tekrar sorulacak.");
        }

        function getAuthHeaders(extraHeaders = {}) {
            const token = getApiToken();

            return {
                ...extraHeaders,
                "Authorization": `Bearer ${token}`
            };
        }

        async function authFetch(url, options = {}, retry = true) {
            options.headers = {
                ...(options.headers || {}),
                "Authorization": `Bearer ${getApiToken()}`
            };

            const response = await fetch(url, options);

            if (response.status === 401 && retry) {
                localStorage.removeItem("API_TOKEN");
                alert("Token hatalı veya eksik. Lütfen tekrar gir.");
                return authFetch(url, options, false);
            }

            return response;
        }

        function addMessage(text, sender, isError = false) {
            const chatBox = document.getElementById("chatBox");
            const div = document.createElement("div");

            div.className = "message " + sender + (isError ? " error" : "");
            div.textContent = text;

            chatBox.appendChild(div);
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        function handleEnter(event) {
            if (event.key === "Enter") {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById("messageInput");
            const message = input.value.trim();

            if (!message) {
                return;
            }

            addMessage(message, "user");
            input.value = "";

            try {
                const response = await authFetch("/chat", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        user_id: 1,
                        username: "web_user",
                        first_name: "Web",
                        message: message
                    })
                });

                const data = await response.json();

                if (!response.ok || data.success === false) {
                    addMessage(data.detail || data.answer || "Hata oluştu.", "bot", true);
                    return;
                }

                addMessage(data.answer || "Cevap alınamadı.", "bot");

            } catch (error) {
                addMessage("İstek atılırken hata oluştu: " + error, "bot", true);
            }
        }

        async function sendVoice() {
            const fileInput = document.getElementById("voiceInput");
            const file = fileInput.files[0];

            if (!file) {
                alert("Önce ses dosyası seç.");
                return;
            }

            addMessage("[Ses dosyası gönderildi]", "user");

            const formData = new FormData();
            formData.append("audio", file);
            formData.append("user_id", "1");
            formData.append("username", "web_user");
            formData.append("first_name", "Web");
            formData.append("language", "tr-TR");

            try {
                const response = await authFetch("/voice", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || data.success === false) {
                    addMessage(data.detail || data.answer || "Ses işlenemedi.", "bot", true);
                    return;
                }

                if (data.transcript) {
                    addMessage("Transcript: " + data.transcript, "bot");
                }

                addMessage(data.answer || "Cevap alınamadı.", "bot");

            } catch (error) {
                addMessage("Ses isteğinde hata oluştu: " + error, "bot", true);
            }
        }

        async function sendImage() {
            const fileInput = document.getElementById("imageInput");
            const captionInput = document.getElementById("captionInput");
            const file = fileInput.files[0];

            if (!file) {
                alert("Önce fotoğraf seç.");
                return;
            }

            addMessage("[Fotoğraf gönderildi] " + captionInput.value, "user");

            const formData = new FormData();
            formData.append("image", file);
            formData.append("user_id", "1");
            formData.append("username", "web_user");
            formData.append("first_name", "Web");
            formData.append("caption", captionInput.value || "");

            try {
                const response = await authFetch("/image", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || data.success === false) {
                    addMessage(data.detail || data.answer || "Fotoğraf işlenemedi.", "bot", true);
                    return;
                }

                addMessage(data.answer || "Cevap alınamadı.", "bot");

            } catch (error) {
                addMessage("Fotoğraf isteğinde hata oluştu: " + error, "bot", true);
            }
        }

        addMessage("Merhaba! Devam etmek için API token girmen gerekebilir.", "bot");
    </script>
</body>
</html>
        """
    )


@app.get("/admin", response_class=HTMLResponse)
@app.get("/admin/", response_class=HTMLResponse)
def admin_page():
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>AI Daily Bot Admin Panel</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f4f7;
            margin: 0;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: auto;
            background: white;
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
        }

        button {
            border: none;
            background: #222;
            color: white;
            padding: 10px 14px;
            border-radius: 9px;
            cursor: pointer;
            margin-right: 8px;
        }

        button.danger {
            background: #b42318;
        }

        .table-block {
            margin-top: 24px;
            border: 1px solid #eee;
            border-radius: 12px;
            padding: 12px;
            overflow-x: auto;
        }

        table {
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
        }

        th, td {
            border: 1px solid #eee;
            padding: 8px;
            vertical-align: top;
            max-width: 360px;
            word-break: break-word;
        }

        th {
            background: #f7f7f7;
            text-align: left;
        }

        .small {
            font-size: 13px;
            color: #666;
        }

        .error {
            background: #fee4e2;
            color: #7a271a;
            padding: 12px;
            border-radius: 10px;
            margin-top: 12px;
        }

        pre {
            white-space: pre-wrap;
            background: #fafafa;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #eee;
        }
    </style>
</head>

<body>
    <div class="container">
        <h1>AI Daily Bot Admin Panel</h1>

        <p class="small">
            Bu panel /admin-data endpointinden verileri çeker. API_AUTH_ENABLED=true ise token zorunludur.
        </p>

        <button onclick="loadAdminData()">Verileri Yenile</button>
        <button class="danger" onclick="resetApiToken()">Token Sıfırla</button>
        <a href="/web" target="_blank"><button>Web Chat</button></a>

        <div id="status"></div>
        <div id="content"></div>
    </div>

    <script>
        function getApiToken() {
            let token = localStorage.getItem("API_TOKEN");

            if (!token) {
                token = prompt("API token giriniz:");

                if (token) {
                    localStorage.setItem("API_TOKEN", token);
                }
            }

            return token || "";
        }

        function resetApiToken() {
            localStorage.removeItem("API_TOKEN");
            alert("Token sıfırlandı.");
        }

        async function authFetch(url, options = {}, retry = true) {
            options.headers = {
                ...(options.headers || {}),
                "Authorization": `Bearer ${getApiToken()}`
            };

            const response = await fetch(url, options);

            if (response.status === 401 && retry) {
                localStorage.removeItem("API_TOKEN");
                alert("Token hatalı veya eksik. Lütfen tekrar gir.");
                return authFetch(url, options, false);
            }

            return response;
        }

        function escapeHtml(value) {
            if (value === null || value === undefined) {
                return "";
            }

            return String(value)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");
        }

        function renderTable(tableName, rows) {
            if (!rows || rows.length === 0) {
                return `
                    <div class="table-block">
                        <h2>${escapeHtml(tableName)}</h2>
                        <p class="small">Kayıt yok.</p>
                    </div>
                `;
            }

            const columns = Object.keys(rows[0]);

            let html = `
                <div class="table-block">
                    <h2>${escapeHtml(tableName)} <span class="small">(${rows.length} kayıt)</span></h2>
                    <table>
                        <thead>
                            <tr>
            `;

            for (const column of columns) {
                html += `<th>${escapeHtml(column)}</th>`;
            }

            html += `
                            </tr>
                        </thead>
                        <tbody>
            `;

            for (const row of rows) {
                html += "<tr>";

                for (const column of columns) {
                    let value = row[column];

                    if (typeof value === "object") {
                        value = JSON.stringify(value, null, 2);
                    }

                    html += `<td>${escapeHtml(value)}</td>`;
                }

                html += "</tr>";
            }

            html += `
                        </tbody>
                    </table>
                </div>
            `;

            return html;
        }

        async function loadAdminData() {
            const status = document.getElementById("status");
            const content = document.getElementById("content");

            status.innerHTML = "<p>Veriler yükleniyor...</p>";
            content.innerHTML = "";

            try {
                const response = await authFetch("/admin-data");
                const data = await response.json();

                if (!response.ok || data.success === false) {
                    status.innerHTML = `<div class="error">${escapeHtml(data.detail || "Admin data alınamadı.")}</div>`;
                    return;
                }

                status.innerHTML = `
                    <p class="small">
                        Database: ${escapeHtml(data.database_path)} |
                        Tablolar: ${escapeHtml((data.table_names || []).join(", "))}
                    </p>
                `;

                let html = "";

                const tables = data.tables || {};

                for (const tableName of Object.keys(tables)) {
                    html += renderTable(tableName, tables[tableName]);
                }

                if (!html) {
                    html = "<p>Gösterilecek tablo yok.</p>";
                }

                content.innerHTML = html;

            } catch (error) {
                status.innerHTML = `<div class="error">Hata: ${escapeHtml(error)}</div>`;
            }
        }

        loadAdminData();
    </script>
</body>
</html>
        """
    )