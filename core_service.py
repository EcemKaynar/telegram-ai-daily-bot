import json
import os
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from pathlib import Path


# -------------------------------------------------------------------
# ENV LOADER
# -------------------------------------------------------------------

def load_env_file():
    env_path = Path(".env")

    if not env_path.exists():
        return

    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            if key and key not in os.environ:
                os.environ[key] = value

    except Exception as error:
        print(f".env okunamadı: {error}")


load_env_file()


# -------------------------------------------------------------------
# OPTIONAL IMPORTS
# -------------------------------------------------------------------

try:
    from openrouter_client import ask_openrouter_with_rag
except Exception as error:
    ask_openrouter_with_rag = None
    print(f"ask_openrouter_with_rag import edilemedi: {error}")

try:
    from openrouter_client import send_chat_request
except Exception as error:
    send_chat_request = None
    print(f"send_chat_request import edilemedi: {error}")

try:
    from rag_utils import search_knowledge_base, format_rag_context
except Exception as error:
    search_knowledge_base = None
    format_rag_context = None
    print(f"rag_utils import edilemedi: {error}")

try:
    from qdrant_rag_tool import (
        search_qdrant_knowledge_base,
        format_qdrant_rag_context,
        get_qdrant_sources_json
    )
except Exception as error:
    search_qdrant_knowledge_base = None
    format_qdrant_rag_context = None
    get_qdrant_sources_json = None
    print(f"qdrant_rag_tool import edilemedi: {error}")


# -------------------------------------------------------------------
# SIMPLE IN-MEMORY SESSION FALLBACK
# -------------------------------------------------------------------

LOCAL_HISTORY = {}
WAITING_FOR_WEATHER_CITY = {}


# -------------------------------------------------------------------
# GENERAL UTILS
# -------------------------------------------------------------------

def run_with_timeout(func, timeout_seconds, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)

    except FutureTimeoutError:
        raise TimeoutError(f"İşlem {timeout_seconds} saniye içinde tamamlanamadı.")

    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def normalize_basic_text(text):
    text = str(text or "").lower().strip()

    replacements = {
        "ı": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def detect_language(text):
    raw = str(text or "")
    normalized = normalize_basic_text(raw)

    turkish_markers = [
        "nasıl",
        "nasil",
        "ne yapmalıyım",
        "ne yapmaliyim",
        "bugün",
        "bugun",
        "yemek",
        "hava",
        "planlama",
        "odaklanamiyorum",
        "olur mu",
        "miyim",
        "misin",
        "lütfen",
        "lutfen"
    ]

    english_markers = [
        "can you",
        "could you",
        "daily plan",
        "what should",
        "how can",
        "please",
        "routine",
        "schedule"
    ]

    if any(marker in normalized for marker in turkish_markers):
        return "tr"

    if any(marker in normalized for marker in english_markers):
        return "en"

    if any(char in raw for char in ["ç", "ğ", "ı", "ö", "ş", "ü", "Ç", "Ğ", "İ", "Ö", "Ş", "Ü"]):
        return "tr"

    return "tr"


def clean_final_answer(answer):
    if answer is None:
        return ""

    answer = str(answer)

    bad_tokens = [
        "<pad>",
        "<PAD>",
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|eot_id|>"
    ]

    for token in bad_tokens:
        answer = answer.replace(token, "")

    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]+", " ", answer)

    # Önceki model çıktılarında gereksiz bold kalıntıları oluyordu.
    answer = answer.replace("**", "")

    return answer.strip()


def is_bad_ai_answer(answer):
    if answer is None:
        return True

    text = str(answer).strip()

    if not text:
        return True

    normalized = normalize_basic_text(text)

    bad_phrases = [
        "<pad>",
        "openrouter response icinde choices bulunamadi",
        "openrouter response içinde choices bulunamadı",
        "cevap alinamadi",
        "cevap alınamadı",
        "su an yapay zeka modeli duzgun bir cevap uretemedi",
        "şu an yapay zeka modeli düzgün bir cevap üretemedi",
        "bilgi tabaninda ilgili bilgi buldum fakat",
        "bilgi tabanında ilgili bilgi buldum fakat",
        "according to the relevant parts of my knowledge base",
        "rag threshold",
        "chunk score",
        "embedding score",
        "middleware",
        "retrieval score"
    ]

    if any(phrase in normalized for phrase in bad_phrases):
        return True

    if len(text) < 2:
        return True

    return False


def build_local_fallback_answer(user_text, language="tr"):
    normalized = normalize_basic_text(user_text)

    if language == "en":
        if "daily plan" in normalized or "schedule" in normalized:
            return (
                "Sure — here is a simple daily plan you can use:\n\n"
                "Morning:\n"
                "- Choose your top 3 priorities.\n"
                "- Start with the most important task.\n"
                "- Use one focused work block before checking messages.\n\n"
                "Midday:\n"
                "- Handle smaller tasks, messages or errands.\n"
                "- Take a short break and reset your focus.\n\n"
                "Afternoon:\n"
                "- Do one more focused work session.\n"
                "- Leave buffer time for unexpected tasks.\n\n"
                "Evening:\n"
                "- Review what you completed.\n"
                "- Move unfinished tasks to tomorrow.\n"
                "- Pick one main goal for the next day."
            )

        return (
            "I can help with that. Start with one small, clear step instead of trying to solve everything at once. "
            "Choose the most important thing, work on it for 20–25 minutes, then take a short break."
        )

    if "odak" in normalized or "verimsiz" in normalized or "calisamiyorum" in normalized:
        return (
            "Odaklanamıyorsan önce kendini zorlamak yerine işi küçült. Şöyle deneyebilirsin:\n\n"
            "1. Yapman gereken işi tek cümleyle yaz.\n"
            "2. Sadece ilk 10 dakikalık kısmı seç.\n"
            "3. Telefonu ve dikkat dağıtan şeyleri uzaklaştır.\n"
            "4. 25 dakika çalış, 5 dakika mola ver.\n"
            "5. Moladan önce dönüşte yapacağın ilk küçük adımı not al.\n\n"
            "Bugün çok verimsiz hissediyorsan hedefi küçült: tek ana görev + kısa çalışma bloğu yeterli olabilir."
        )

    if "yemek" in normalized or "ne yesem" in normalized or "ne yiyebilirim" in normalized:
        return (
            "Bugün basit ve doyurucu bir şey istiyorsan şu seçeneklerden biri iyi olur:\n\n"
            "- Tost + ayran\n"
            "- Omlet + ekmek + domates/salatalık\n"
            "- Makarna + yoğurt\n"
            "- Mercimek çorbası + ekmek\n"
            "- Yoğurtlu yulaf veya pratik kahvaltı tabağı\n\n"
            "Enerjin düşükse mükemmel yemek yapmaya çalışma; doyurucu ve kolay bir seçenek seçmen yeterli."
        )

    if "bitcoin" in normalized or "kripto" in normalized or "yatirim" in normalized:
        return (
            "Bu konuda doğrudan “al” ya da “alma” şeklinde yatırım tavsiyesi veremem. "
            "Bitcoin ve kripto varlıklar yüksek dalgalanma ve risk içerir. Karar vermeden önce risk toleransını, "
            "bütçeni, kaybetmeyi göze alabileceğin tutarı ve güvenilir uzman görüşlerini değerlendirmen daha doğru olur."
        )

    return (
        "Bunu daha yönetilebilir hale getirmek için önce küçük bir adım seçebilirsin. "
        "Yapman gereken şeyi netleştir, en önemli kısmı belirle ve 20–25 dakikalık kısa bir başlangıç yap."
    )


# -------------------------------------------------------------------
# HISTORY HELPERS
# -------------------------------------------------------------------

def get_history_context(user_id, limit=6):
    user_key = str(user_id)

    try:
        import history_manager

        possible_functions = [
            "get_history_context",
            "get_recent_history_context",
            "get_conversation_context",
            "get_recent_messages_text"
        ]

        for function_name in possible_functions:
            function = getattr(history_manager, function_name, None)

            if not function:
                continue

            try:
                return function(user_id=user_id, limit=limit)
            except TypeError:
                try:
                    return function(user_id, limit)
                except TypeError:
                    return function(user_id)

    except Exception:
        pass

    history = LOCAL_HISTORY.get(user_key, [])[-limit:]

    if not history:
        return ""

    lines = []

    for item in history:
        user_message = item.get("user", "")
        assistant_message = item.get("assistant", "")

        if user_message:
            lines.append(f"User: {user_message}")

        if assistant_message:
            lines.append(f"Assistant: {assistant_message}")

    return "\n".join(lines).strip()


def save_history_context(user_id, user_text, answer):
    user_key = str(user_id)

    try:
        import history_manager

        possible_functions = [
            "save_interaction",
            "add_interaction",
            "append_history",
            "save_message_pair",
            "add_message_pair"
        ]

        for function_name in possible_functions:
            function = getattr(history_manager, function_name, None)

            if not function:
                continue

            try:
                function(user_id=user_id, user_text=user_text, answer=answer)
                return
            except TypeError:
                try:
                    function(user_id, user_text, answer)
                    return
                except TypeError:
                    continue

    except Exception:
        pass

    LOCAL_HISTORY.setdefault(user_key, []).append({
        "user": user_text,
        "assistant": answer,
        "created_at": datetime.now().isoformat()
    })

    LOCAL_HISTORY[user_key] = LOCAL_HISTORY[user_key][-20:]


# -------------------------------------------------------------------
# DATABASE LOGGING
# -------------------------------------------------------------------

def log_interaction_to_db(
    user_id,
    username,
    first_name,
    user_text,
    answer,
    source_type,
    sources
):
    sources_json = json.dumps(sources or [], ensure_ascii=False)

    try:
        import database

        payload = {
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "question": user_text,
            "message": user_text,
            "user_message": user_text,
            "user_text": user_text,
            "answer": answer,
            "bot_answer": answer,
            "source_type": source_type,
            "sources": sources_json,
            "sources_json": sources_json
        }

        possible_functions = [
            "save_interaction",
            "log_interaction",
            "insert_interaction",
            "record_interaction",
            "add_interaction"
        ]

        for function_name in possible_functions:
            function = getattr(database, function_name, None)

            if not function:
                continue

            try:
                function(**payload)
                print("Interaction DB kaydı yapıldı.")
                return True
            except Exception:
                pass

            try:
                function(
                    user_id,
                    username,
                    first_name,
                    user_text,
                    answer,
                    source_type,
                    sources_json
                )
                print("Interaction DB kaydı yapıldı.")
                return True
            except Exception:
                pass

            try:
                function(user_id, user_text, answer)
                print("Interaction DB kaydı yapıldı.")
                return True
            except Exception:
                pass

    except Exception as error:
        print(f"database.py üzerinden kayıt denenemedi: {error}")

    try:
        import sqlite3

        db_path = os.getenv("DATABASE_PATH", "bot_messages.db")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT,
                username TEXT,
                first_name TEXT,
                question TEXT,
                answer TEXT,
                created_at TEXT
            )
        """)

        cursor.execute("PRAGMA table_info(interactions)")
        columns_info = cursor.fetchall()
        existing_columns = [column[1] for column in columns_info]

        row_data = {}

        if "session_id" in existing_columns:
            row_data["session_id"] = f"{user_id}_web_session"

        if "user_id" in existing_columns:
            row_data["user_id"] = str(user_id)

        if "username" in existing_columns:
            row_data["username"] = username or ""

        if "first_name" in existing_columns:
            row_data["first_name"] = first_name or ""

        if "question" in existing_columns:
            row_data["question"] = user_text or ""

        if "user_message" in existing_columns:
            row_data["user_message"] = user_text or ""

        if "message" in existing_columns:
            row_data["message"] = user_text or ""

        if "answer" in existing_columns:
            row_data["answer"] = answer or ""

        if "bot_answer" in existing_columns:
            row_data["bot_answer"] = answer or ""

        if "source_type" in existing_columns:
            row_data["source_type"] = source_type or ""

        if "sources_json" in existing_columns:
            row_data["sources_json"] = sources_json

        if "sources" in existing_columns:
            row_data["sources"] = sources_json

        if "created_at" in existing_columns:
            row_data["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if not row_data:
            raise RuntimeError("interactions tablosunda yazılabilecek uygun kolon bulunamadı.")

        columns = list(row_data.keys())
        placeholders = ", ".join(["?"] * len(columns))
        column_sql = ", ".join(columns)
        values = [row_data[column] for column in columns]

        cursor.execute(
            f"INSERT INTO interactions ({column_sql}) VALUES ({placeholders})",
            values
        )

        conn.commit()
        conn.close()

        print("Interaction DB kaydı yapıldı. Mevcut tablo kolonlarına göre SQLite kullanıldı.")
        return True

    except Exception as error:
        print(f"Interaction DB kaydı yapılamadı: {error}")
        return False

# -------------------------------------------------------------------
# BASIC CONVERSATION
# -------------------------------------------------------------------

def get_basic_conversation_answer(user_text, language="tr"):
    text = normalize_basic_text(user_text)

    greetings = [
        "merhaba",
        "selam",
        "slm",
        "hello",
        "hi",
        "hey"
    ]

    how_are_you = [
        "nasilsin",
        "nasilsin?",
        "naber",
        "ne haber",
        "how are you"
    ]

    thanks = [
        "tesekkurler",
        "tesekkur ederim",
        "sag ol",
        "sağ ol",
        "thanks",
        "thank you"
    ]

    if text in greetings:
        if language == "en":
            return "Hi! How can I help you today?"
        return "Merhaba! Bugün sana nasıl yardımcı olayım?"

    if text in how_are_you:
        if language == "en":
            return "I'm good, thank you. How can I help you today?"
        return "İyiyim, teşekkür ederim. Bugün sana nasıl yardımcı olayım?"

    if text in thanks:
        if language == "en":
            return "You're welcome!"
        return "Rica ederim!"

    return None


# -------------------------------------------------------------------
# WEATHER TOOL FLOW
# -------------------------------------------------------------------

def looks_like_weather_question(user_text):
    text = normalize_basic_text(user_text)

    direct_weather_words = [
        "hava",
        "hava durumu",
        "yagmur",
        "yagacak",
        "semsiye",
        "sicaklik",
        "derece",
        "ruzgar",
        "kar",
        "firtina",
        "forecast",
        "weather",
        "rain",
        "umbrella",
        "temperature",
        "wind",
        "snow",
        "storm"
    ]

    if any(word in text for word in direct_weather_words):
        return True

    clothing_weather_patterns = [
        "ne giysem",
        "ne giymeliyim",
        "mont gerekir mi",
        "ceket gerekir mi",
        "usur muyum",
        "usurum",
        "soguk mu",
        "sicak mi"
    ]

    if any(pattern in text for pattern in clothing_weather_patterns):
        return True

    return False


def clean_extracted_city(city):
    city = str(city or "").strip()

    city = re.sub(
        r"\b(hava|durumu|nasil|nasıl|bugun|bugün|yarin|yarın|derece|sicaklik|sıcaklık|yagmur|yağmur|var mi|var mı)\b",
        " ",
        city,
        flags=re.IGNORECASE
    )

    city = city.replace("'", " ")
    city = re.sub(r"\s+", " ", city)

    return city.strip(" .,!?:;")


def extract_city_from_weather_sentence(user_text):
    raw_text = str(user_text or "").strip()
    text = normalize_basic_text(raw_text)

    unclear_location_phrases = [
        "yasadigim yerde",
        "bulundugum yerde",
        "burada",
        "burasi",
        "konumumda",
        "benim oldugum yerde"
    ]

    if any(phrase in text for phrase in unclear_location_phrases):
        return ""

    patterns = [
        r"^(.+?)(?:'?(?:de|da|te|ta))\s+hava",
        r"^(.+?)\s+hava\s+(?:nasil|nasıl|durumu)",
        r"^(.+?)\s+weather",
        r"weather\s+in\s+(.+)$",
        r"how\s+is\s+the\s+weather\s+in\s+(.+)$"
    ]

    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)

        if match:
            city = clean_extracted_city(match.group(1))

            if city:
                return city

    return ""


def call_weather_tool(city):
    try:
        from tools import weather_tool

        possible_functions = [
            "get_weather",
            "get_weather_info",
            "fetch_weather",
            "get_weather_forecast",
            "weather_tool"
        ]

        last_error = None

        for function_name in possible_functions:
            function = getattr(weather_tool, function_name, None)

            if not function:
                continue

            try:
                return function(city)
            except TypeError as error:
                last_error = error

                try:
                    return function(location=city)
                except TypeError as second_error:
                    last_error = second_error
                    continue

        raise RuntimeError(f"Weather tool fonksiyonu bulunamadı. Son hata: {last_error}")

    except Exception as error:
        print(f"Weather tool hatası: {error}")

        return {
            "success": False,
            "error": str(error),
            "city": city
        }


def format_weather_answer(weather_result, city, language="tr"):
    if isinstance(weather_result, str):
        return weather_result

    if not isinstance(weather_result, dict):
        return str(weather_result)

    if weather_result.get("answer"):
        return str(weather_result["answer"])

    if weather_result.get("summary"):
        return str(weather_result["summary"])

    if weather_result.get("success") is False:
        if language == "en":
            return f"I couldn't retrieve the weather for {city} right now. Please try again later."
        return f"{city} için hava durumunu şu anda alamadım. Biraz sonra tekrar deneyebilir misin?"

    temperature = (
        weather_result.get("temperature")
        or weather_result.get("temp")
        or weather_result.get("current_temperature")
    )

    condition = (
        weather_result.get("condition")
        or weather_result.get("description")
        or weather_result.get("weather")
    )

    rain = (
        weather_result.get("rain")
        or weather_result.get("precipitation")
        or weather_result.get("precipitation_probability")
    )

    parts = []

    if language == "en":
        parts.append(f"Weather for {city}:")
    else:
        parts.append(f"{city} için hava durumu:")

    if temperature is not None:
        parts.append(f"Sıcaklık: {temperature}")

    if condition:
        parts.append(f"Durum: {condition}")

    if rain is not None:
        parts.append(f"Yağış: {rain}")

    if len(parts) == 1:
        return json.dumps(weather_result, ensure_ascii=False, indent=2)

    return "\n".join(parts)


def handle_weather_flow(user_id, user_text, language="tr"):
    user_key = str(user_id)

    if WAITING_FOR_WEATHER_CITY.get(user_key):
        city = clean_extracted_city(user_text)

        if city and len(city) <= 50:
            WAITING_FOR_WEATHER_CITY[user_key] = False

            weather_result = call_weather_tool(city)
            answer = format_weather_answer(weather_result, city, language)

            return {
                "handled": True,
                "answer": answer,
                "source_type": "weather_tool",
                "sources": [
                    {
                        "tool": "weather",
                        "city": city,
                        "result": weather_result
                    }
                ]
            }

    if not looks_like_weather_question(user_text):
        return {
            "handled": False
        }

    city = extract_city_from_weather_sentence(user_text)

    if not city:
        WAITING_FOR_WEATHER_CITY[user_key] = True

        if language == "en":
            answer = "Which city should I check the weather for?"
        else:
            answer = "Hangi şehir için hava durumuna bakmamı istersin?"

        return {
            "handled": True,
            "answer": answer,
            "source_type": "weather_city_needed",
            "sources": []
        }

    weather_result = call_weather_tool(city)
    answer = format_weather_answer(weather_result, city, language)

    return {
        "handled": True,
        "answer": answer,
        "source_type": "weather_tool",
        "sources": [
            {
                "tool": "weather",
                "city": city,
                "result": weather_result
            }
        ]
    }


# -------------------------------------------------------------------
# RAG BACKEND
# -------------------------------------------------------------------

def get_rag_backend():
    return os.getenv("RAG_BACKEND", "json").strip().lower()


def search_rag_backend(user_text):
    backend = get_rag_backend()

    if backend == "qdrant":
        if (
            search_qdrant_knowledge_base
            and format_qdrant_rag_context
            and get_qdrant_sources_json
        ):
            try:
                qdrant_result = search_qdrant_knowledge_base(user_text)

                if qdrant_result.get("found"):
                    rag_context = format_qdrant_rag_context(
                        qdrant_result.get("results", [])
                    )
                    sources_json = get_qdrant_sources_json(
                        qdrant_result.get("results", [])
                    )

                    return {
                        "found": True,
                        "backend": "qdrant",
                        "context": rag_context,
                        "sources": qdrant_result.get("results", []),
                        "sources_json": sources_json,
                        "raw": qdrant_result
                    }

                print("Qdrant sonuç bulamadı, JSON RAG fallback deneniyor.")

            except Exception as error:
                print(f"Qdrant RAG hatası, JSON fallback deneniyor: {error}")

        else:
            print("Qdrant RAG tool import edilemedi, JSON fallback deneniyor.")

    if search_knowledge_base and format_rag_context:
        try:
            json_result = search_knowledge_base(user_text)

            if json_result.get("found"):
                rag_context = format_rag_context(json_result.get("results", []))

                return {
                    "found": True,
                    "backend": "json",
                    "context": rag_context,
                    "sources": json_result.get("results", []),
                    "sources_json": json.dumps(
                        json_result.get("results", []),
                        ensure_ascii=False
                    ),
                    "raw": json_result
                }

        except Exception as error:
            print(f"JSON RAG hatası: {error}")

    return {
        "found": False,
        "backend": backend,
        "context": "",
        "sources": [],
        "sources_json": "[]",
        "raw": {}
    }


def call_rag_llm(user_text, rag_context, history_context):
    if not ask_openrouter_with_rag:
        raise RuntimeError("ask_openrouter_with_rag fonksiyonu yok.")

    try:
        return ask_openrouter_with_rag(
            user_text=user_text,
            rag_context=rag_context,
            history_context=history_context
        )
    except TypeError:
        try:
            return ask_openrouter_with_rag(
                user_text,
                rag_context,
                history_context
            )
        except TypeError:
            return ask_openrouter_with_rag(
                user_text,
                rag_context
            )


# -------------------------------------------------------------------
# GENERAL LLM FALLBACK
# -------------------------------------------------------------------

def generate_general_fallback_answer(user_text, history_context="", language="tr"):
    if send_chat_request:
        if language == "en":
            system_prompt = """
You are a helpful daily-life assistant.
Answer in English.
Be practical, specific and conversational.
Do not mention internal tools, RAG, chunks, embeddings, thresholds, middleware or retrieval.
If the user asks for a plan, routine, checklist or recommendation, create it directly.
"""
        else:
            system_prompt = """
Sen yardımcı bir günlük yaşam asistanısın.
Türkçe cevap ver.
Pratik, net ve uygulanabilir öneriler sun.
İç sistem detaylarından, RAG, chunk, embedding, threshold, middleware veya retrieval gibi kelimelerden bahsetme.
Kullanıcı plan, rutin, liste veya öneri isterse doğrudan oluştur.
"""

        user_prompt = f"""
Conversation history:
{history_context or "No previous conversation context."}

User message:
{user_text}

Write the best possible answer.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": user_prompt.strip()
            }
        ]

        try:
            answer = run_with_timeout(
                send_chat_request,
                75,
                messages
            )

            answer = clean_final_answer(answer)

            if not is_bad_ai_answer(answer):
                return answer

        except Exception as error:
            print(f"General fallback LLM hatası: {error}")

    return build_local_fallback_answer(user_text, language)


# -------------------------------------------------------------------
# MAIN TEXT FLOW
# -------------------------------------------------------------------

def handle_text_message(user_id, username, first_name, user_text):
    language = detect_language(user_text)
    history_context = get_history_context(user_id)

    if not user_text or not str(user_text).strip():
        if language == "en":
            return {
                "answer": "Please write a message so I can help.",
                "source_type": "empty_message",
                "sources": []
            }

        return {
            "answer": "Sana yardımcı olabilmem için bir mesaj yazman gerekiyor.",
            "source_type": "empty_message",
            "sources": []
        }

    basic_answer = get_basic_conversation_answer(user_text, language)

    if basic_answer:
        return {
            "answer": basic_answer,
            "source_type": "basic_conversation",
            "sources": []
        }

    weather_flow = handle_weather_flow(user_id, user_text, language)

    if weather_flow.get("handled"):
        return {
            "answer": weather_flow.get("answer", ""),
            "source_type": weather_flow.get("source_type", "weather"),
            "sources": weather_flow.get("sources", [])
        }

    rag_search = search_rag_backend(user_text)

    if rag_search.get("found"):
        rag_context = rag_search.get("context", "")

        try:
            ai_answer = run_with_timeout(
                call_rag_llm,
                75,
                user_text,
                rag_context,
                history_context
            )

            ai_answer = clean_final_answer(ai_answer)

            if is_bad_ai_answer(ai_answer):
                raise RuntimeError("RAG AI cevabı kötü veya boş döndü.")

            print(f"RAG cevabı üretildi. Backend: {rag_search.get('backend')}")

            return {
                "answer": ai_answer,
                "source_type": f"{rag_search.get('backend')}_rag",
                "sources": rag_search.get("sources", [])
            }

        except Exception as error:
            print(
                f"RAG AI cevabı üretilemedi. "
                f"Backend: {rag_search.get('backend')}. "
                f"Hata: {error}"
            )

            ai_answer = generate_general_fallback_answer(
                user_text=user_text,
                history_context=history_context,
                language=language
            )

            ai_answer = clean_final_answer(ai_answer)

            return {
                "answer": ai_answer,
                "source_type": f"{rag_search.get('backend')}_rag_ai_failed_general_fallback",
                "sources": rag_search.get("sources", [])
            }

    ai_answer = generate_general_fallback_answer(
        user_text=user_text,
        history_context=history_context,
        language=language
    )

    ai_answer = clean_final_answer(ai_answer)

    return {
        "answer": ai_answer,
        "source_type": "general_fallback",
        "sources": []
    }


def process_text_message(
    user_id,
    username="",
    first_name="",
    message="",
    **kwargs
):
    user_text = message or kwargs.get("text") or kwargs.get("user_text") or ""
    user_text = str(user_text).strip()

    try:
        result = handle_text_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            user_text=user_text
        )

        answer = clean_final_answer(result.get("answer", ""))

        if is_bad_ai_answer(answer):
            language = detect_language(user_text)
            answer = generate_general_fallback_answer(
                user_text=user_text,
                history_context=get_history_context(user_id),
                language=language
            )
            answer = clean_final_answer(answer)

        result["answer"] = answer
        result.setdefault("source_type", "unknown")
        result.setdefault("sources", [])

        save_history_context(user_id, user_text, answer)

        log_interaction_to_db(
            user_id=user_id,
            username=username,
            first_name=first_name,
            user_text=user_text,
            answer=answer,
            source_type=result.get("source_type", "unknown"),
            sources=result.get("sources", [])
        )

        return result

    except Exception as error:
        print("process_text_message genel hata:")
        print(traceback.format_exc())

        language = detect_language(user_text)

        if language == "en":
            answer = "Something went wrong while processing your message. Please try again in a few seconds."
        else:
            answer = "Mesajını işlerken bir hata oluştu. Birkaç saniye sonra tekrar dener misin?"

        result = {
            "answer": answer,
            "source_type": "error",
            "sources": [
                {
                    "error": str(error)
                }
            ]
        }

        log_interaction_to_db(
            user_id=user_id,
            username=username,
            first_name=first_name,
            user_text=user_text,
            answer=answer,
            source_type="error",
            sources=result["sources"]
        )

        return result


# -------------------------------------------------------------------
# VOICE / IMAGE COMPATIBILITY
# -------------------------------------------------------------------

def process_voice_message(
    user_id,
    username="",
    first_name="",
    audio_path=None,
    message="",
    **kwargs
):
    transcript = (
        kwargs.get("transcript")
        or kwargs.get("text")
        or message
        or ""
    )

    if transcript:
        result = process_text_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            message=transcript
        )
        result["transcript"] = transcript
        result["source_type"] = f"voice_{result.get('source_type', 'unknown')}"
        return result

    try:
        from openrouter_client import transcribe_audio

        transcript = transcribe_audio(audio_path)

        result = process_text_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            message=transcript
        )

        result["transcript"] = transcript
        result["source_type"] = f"voice_{result.get('source_type', 'unknown')}"

        return result

    except Exception as error:
        print(f"Voice processing hatası: {error}")

        return {
            "answer": "Ses mesajını şu anda işleyemedim. İstersen metin olarak yazabilirsin.",
            "source_type": "voice_error",
            "sources": [
                {
                    "error": str(error)
                }
            ]
        }


def process_image_message(
    user_id,
    username="",
    first_name="",
    image_path=None,
    caption="",
    message="",
    **kwargs
):
    user_text = caption or message or kwargs.get("text") or ""

    try:
        from openrouter_client import analyze_image

        answer = analyze_image(
            image_path=image_path,
            user_text=user_text
        )

        answer = clean_final_answer(answer)

        if is_bad_ai_answer(answer):
            raise RuntimeError("Image AI cevabı kötü veya boş döndü.")

        log_interaction_to_db(
            user_id=user_id,
            username=username,
            first_name=first_name,
            user_text=user_text or "[image]",
            answer=answer,
            source_type="image_vision",
            sources=[]
        )

        return {
            "answer": answer,
            "source_type": "image_vision",
            "sources": []
        }

    except Exception as error:
        print(f"Image processing hatası: {error}")

        fallback_text = user_text or "Bu görsel hakkında yardımcı olur musun?"

        result = process_text_message(
            user_id=user_id,
            username=username,
            first_name=first_name,
            message=fallback_text
        )

        result["source_type"] = f"image_fallback_{result.get('source_type', 'unknown')}"

        return result


# Eski api_server importları bozulmasın diye aliaslar
def process_message(*args, **kwargs):
    return process_text_message(*args, **kwargs)


def handle_message(*args, **kwargs):
    return process_text_message(*args, **kwargs)