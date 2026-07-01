import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import telebot
from telebot import types
from flask import Flask, request
from dotenv import load_dotenv

from openrouter_client import (
    ask_openrouter_with_context,
    ask_openrouter_with_tool_result,
    analyze_image_with_openrouter,
    ask_openrouter_with_rag
)

from tool_router import detect_tool_call
from tools.weather_tool import get_weather_by_city, to_json_text
from database import init_db, save_interaction, save_service_log, save_rag_log
from history_manager import get_session_and_history, refresh_session_summary_if_needed
from voice_utils import speech_to_text, text_to_speech
from rag_utils import (
    search_knowledge_base,
    format_rag_context,
    get_rag_sources_json,
    build_direct_rag_answer
)


load_dotenv(".env")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "telegram-ai-bot-secret")


if not TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN bulunamadı. .env dosyasını kontrol et.")

if not WEBHOOK_URL:
    raise ValueError("WEBHOOK_URL bulunamadı. Ngrok linkini .env dosyasına eklemelisin.")


init_db()

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


def is_bad_ai_answer(ai_answer):
    if ai_answer is None:
        return True

    if not isinstance(ai_answer, str):
        return True

    cleaned = ai_answer.strip()

    return cleaned in ["", "[]", "{}", "null", "None", "none", "undefined"]


def call_with_timeout(func, timeout_seconds=15):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)

    try:
        result = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False, cancel_futures=True)
        return result, None

    except TimeoutError:
        executor.shutdown(wait=False, cancel_futures=True)
        return None, "timeout"

    except Exception as error:
        executor.shutdown(wait=False, cancel_futures=True)
        return None, str(error)


def looks_like_weather_question(user_text):
    text = user_text.lower()

    weather_words = [
        "hava",
        "yağmur",
        "şemsiye",
        "sıcaklık",
        "soğuk",
        "rüzgar",
        "mont",
        "ceket",
        "üşür",
        "üşürüm",
        "giymeliyim",
        "giysem",
        "dışarı",
        "weather",
        "rain",
        "umbrella",
        "temperature",
        "wind"
    ]

    return any(word in text for word in weather_words)


def generate_ai_answer(user_text, session_id, history_context, user_id, username):
    print("generate_ai_answer çalıştı.")
    print(f"AI'ye giden mesaj: {user_text}")

    if not looks_like_weather_question(user_text):
        print("Hava durumu sorusu değil. Önce RAG çalışacak.")

        rag_search = search_knowledge_base(user_text)

        print("RAG araması yapıldı.")
        print(rag_search)

        if rag_search["found"]:
            rag_context = format_rag_context(rag_search["results"])
            rag_sources = get_rag_sources_json(rag_search["results"])

            print("RAG sonucu bulundu.")
            print(rag_sources)

            try:
                save_rag_log(
                    session_id=session_id,
                    user_id=user_id,
                    username=username,
                    question=user_text,
                    source_files=rag_sources,
                    matched_text=rag_context
                )
                print("RAG log kaydedildi.")
            except Exception as error:
                print(f"RAG log kayıt hatası: {error}")

            ai_answer, rag_ai_error = call_with_timeout(
                func=lambda: ask_openrouter_with_rag(
                    user_text=user_text,
                    rag_context=rag_context,
                    history_context=history_context
                ),
                timeout_seconds=15
            )

            if rag_ai_error or is_bad_ai_answer(ai_answer):
                print(f"RAG AI cevabı üretilemedi, direkt bilgi tabanı cevabına düşüldü: {rag_ai_error}")

                ai_answer = build_direct_rag_answer(
                    question=user_text,
                    rag_results=rag_search["results"]
                )

                print("RAG cevabı direkt bilgi tabanından üretildi.")
                print(ai_answer)

                return ai_answer

            ai_answer = f"{ai_answer}\n\nKaynak: {rag_search['results'][0]['source_file']}"

            print("RAG cevabı AI ile üretildi.")
            print(ai_answer)

            return ai_answer

        print("RAG sonucu bulunamadı.")

        return (
            "Bu konuda bilgi tabanımda yeterli bilgi bulamadım. "
            "Şu an yalnızca günlük planlama, odaklanma, mola yönetimi, plan defteri, günlük rutin, "
            "basit yemek fikirleri ve hava durumuna göre hazırlık konularındaki bilgi tabanıma göre cevap verebilirim."
        )

    print("Hava durumu sorusu algılandı. Tool calling çalışacak.")

    tool_call = detect_tool_call(user_text, history_context)

    print("Tool call sonucu:")
    print(tool_call)

    if tool_call.get("tool") == "weather":
        city = tool_call.get("city")

        if not city:
            return "Hava durumunu kontrol edebilmem için şehir adını yazar mısın?"

        weather_result = get_weather_by_city(city)

        for log in weather_result.get("logs", []):
            save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name="open_meteo_weather",
                request_data=to_json_text(log.get("request")),
                response_data=to_json_text(log.get("response"))
            )

        if weather_result["success"]:
            weather_context = to_json_text(weather_result["data"])

            if tool_call.get("raw_tool_call") and tool_call.get("assistant_message"):
                ai_answer = ask_openrouter_with_tool_result(
                    user_text=user_text,
                    assistant_message=tool_call["assistant_message"],
                    tool_call=tool_call["raw_tool_call"],
                    tool_result_text=weather_context,
                    history_context=history_context
                )
            else:
                ai_answer = ask_openrouter_with_context(
                    user_text=user_text,
                    context_text=weather_context,
                    history_context=history_context
                )

            return ai_answer

        return weather_result["message"]

    return "Hava durumunu kontrol edebilmem için şehir adını da yazar mısın?"


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


@bot.message_handler(content_types=["voice"])
def handle_voice(message):
    print("VOICE handler çalıştı.")

    user_id, username, first_name = get_user_info_from_message(message)

    try:
        session_id, history_context = get_session_and_history(
            user_id=user_id,
            username=username
        )
        print("Session/history alındı.")
    except Exception as error:
        print(f"Session/history hatası: {error}")
        session_id = f"{user_id}_temporary"
        history_context = ""

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

        stt_result = speech_to_text(
            audio_path=temp_voice_path,
            language="tr-TR"
        )

        if not stt_result["success"]:
            bot.reply_to(message, stt_result["message"])

            save_interaction(
                session_id=session_id,
                user_id=user_id,
                username=username,
                first_name=first_name,
                question="[VOICE_STT_FAILED]",
                answer=stt_result["message"]
            )

            return

        transcript = stt_result["text"]

        print(f"Sesli mesaj metni: {transcript}")

        ai_answer = generate_ai_answer(
            user_text=transcript,
            session_id=session_id,
            history_context=history_context,
            user_id=user_id,
            username=username
        )

        save_interaction(
            session_id=session_id,
            user_id=user_id,
            username=username,
            first_name=first_name,
            question=f"[VOICE] {transcript}",
            answer=ai_answer
        )

        text_reply = (
            f"Sesli mesajını şöyle anladım:\n"
            f"{transcript}\n\n"
            f"Cevabım:\n"
            f"{ai_answer}"
        )

        bot.reply_to(message, text_reply)

        tts_result = text_to_speech(
            text=ai_answer,
            language="tr"
        )

        if tts_result["success"]:
            tts_voice_path = tts_result["voice_path"]

            with open(tts_voice_path, "rb") as voice_file:
                bot.send_voice(
                    chat_id=message.chat.id,
                    voice=voice_file,
                    caption="Sesli cevap",
                    reply_to_message_id=message.message_id
                )

    except Exception as error:
        print(f"Sesli mesaj işleme hatası: {error}")
        bot.reply_to(message, "Sesli mesajı işlerken bir hata oluştu. Lütfen tekrar dener misin?")

    finally:
        if temp_voice_path and os.path.exists(temp_voice_path):
            os.remove(temp_voice_path)

        if tts_voice_path and os.path.exists(tts_voice_path):
            os.remove(tts_voice_path)


@bot.message_handler(content_types=["photo"])
def handle_photo(message):
    print("PHOTO handler çalıştı.")

    user_id, username, first_name = get_user_info_from_message(message)

    try:
        session_id, history_context = get_session_and_history(
            user_id=user_id,
            username=username
        )
        print("Session/history alındı.")
    except Exception as error:
        print(f"Session/history hatası: {error}")
        session_id = f"{user_id}_temporary"
        history_context = ""

    bot.send_chat_action(message.chat.id, "typing")

    caption = message.caption or ""
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

        ai_answer = analyze_image_with_openrouter(
            image_path=temp_image_path,
            caption=caption,
            history_context=history_context
        )

    except Exception as error:
        print(f"Fotoğraf analiz hatası: {error}")
        ai_answer = "Fotoğrafı analiz ederken bir hata oluştu. Lütfen tekrar dener misin?"

    finally:
        if temp_image_path and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    try:
        save_interaction(
            session_id=session_id,
            user_id=user_id,
            username=username,
            first_name=first_name,
            question=f"[PHOTO] {caption}",
            answer=ai_answer
        )
        print("DB kaydı yapıldı.")
    except Exception as error:
        print(f"DB kayıt hatası: {error}")

    # Free model limitlerini hızlı tüketmemesi için şimdilik kapalı.
    # refresh_session_summary_if_needed(session_id)

    bot.reply_to(message, ai_answer)


@bot.message_handler(func=lambda message: True, content_types=["text"])
def handle_message(message):
    print("TEXT handler çalıştı.")

    user_text = message.text
    print(f"Gelen text mesaj: {user_text}")

    user_id, username, first_name = get_user_info_from_message(message)

    try:
        session_id, history_context = get_session_and_history(
            user_id=user_id,
            username=username
        )
        print("Session/history alındı.")
    except Exception as error:
        print(f"Session/history hatası: {error}")
        session_id = f"{user_id}_temporary"
        history_context = ""

    try:
        bot.send_chat_action(message.chat.id, "typing")
        print("Typing action gönderildi.")
    except Exception as error:
        print(f"Typing action hatası: {error}")

    print("generate_ai_answer çağrılacak.")

    ai_answer = generate_ai_answer(
        user_text=user_text,
        session_id=session_id,
        history_context=history_context,
        user_id=user_id,
        username=username
    )

    print("AI cevap üretildi.")
    print(ai_answer)

    try:
        save_interaction(
            session_id=session_id,
            user_id=user_id,
            username=username,
            first_name=first_name,
            question=user_text,
            answer=ai_answer
        )
        print("DB kaydı yapıldı.")
    except Exception as error:
        print(f"DB kayıt hatası: {error}")

    # Free model limitlerini hızlı tüketmemesi için şimdilik kapalı.
    # refresh_session_summary_if_needed(session_id)

    bot.reply_to(message, ai_answer)


@app.route("/", methods=["GET"])
def index():
    return "Telegram AI Bot webhook server is running."


@app.route(WEBHOOK_PATH, methods=["POST"])
def webhook():
    print("Webhook isteği geldi.")

    if request.headers.get("content-type") == "application/json":
        json_string = request.get_data().decode("utf-8")
        update = types.Update.de_json(json_string)

        print("Telegram update alındı.")
        print(json_string[:500])

        bot.process_new_updates([update])

        return "OK", 200

    return "Unsupported Media Type", 415


if __name__ == "__main__":
    full_webhook_url = WEBHOOK_URL.rstrip("/") + WEBHOOK_PATH

    bot.remove_webhook()
    bot.set_webhook(url=full_webhook_url)

    print("Webhook bot başlatıldı.")
    print(f"Webhook URL: {full_webhook_url}")

    app.run(host="0.0.0.0", port=5000)