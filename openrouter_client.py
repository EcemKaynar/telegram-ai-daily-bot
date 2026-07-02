import base64
import json
import mimetypes
import os
import time
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv


load_dotenv(".env")


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "")
OPENROUTER_TOOL_MODEL = os.getenv("OPENROUTER_TOOL_MODEL", OPENROUTER_MODEL)

MODEL_COOLDOWN_SECONDS = int(os.getenv("MODEL_COOLDOWN_SECONDS", "900"))
MODEL_STATE_FILE = os.getenv("MODEL_STATE_FILE", "model_state.json")

TEMPORARY_ERROR_STATUS_CODES = {429, 500, 502, 503, 529}


def parse_model_pool(env_name, fallback_model):
    value = os.getenv(env_name, "")

    if value:
        models = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

        if models:
            return models

    if fallback_model:
        return [fallback_model]

    return []


CHAT_MODEL_POOL = parse_model_pool(
    env_name="OPENROUTER_CHAT_MODEL_POOL",
    fallback_model=OPENROUTER_MODEL
)

TOOL_MODEL_POOL = parse_model_pool(
    env_name="OPENROUTER_TOOL_MODEL_POOL",
    fallback_model=OPENROUTER_TOOL_MODEL
)

VISION_MODEL_POOL = parse_model_pool(
    env_name="OPENROUTER_VISION_MODEL_POOL",
    fallback_model=OPENROUTER_MODEL
)


def get_now_iso():
    return datetime.now().isoformat()


def load_model_state():
    if not os.path.exists(MODEL_STATE_FILE):
        return {
            "cooldowns": {}
        }

    try:
        with open(MODEL_STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)

    except Exception:
        return {
            "cooldowns": {}
        }


def save_model_state(state):
    try:
        with open(MODEL_STATE_FILE, "w", encoding="utf-8") as file:
            json.dump(state, file, ensure_ascii=False, indent=2)

    except Exception as error:
        print(f"Model state kaydedilemedi: {error}")


def parse_iso_datetime(value):
    if not value:
        return None

    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def is_model_on_cooldown(model):
    state = load_model_state()
    cooldowns = state.get("cooldowns", {})

    cooldown_until_text = cooldowns.get(model)

    if not cooldown_until_text:
        return False

    cooldown_until = parse_iso_datetime(cooldown_until_text)

    if not cooldown_until:
        return False

    if datetime.now() >= cooldown_until:
        cooldowns.pop(model, None)
        state["cooldowns"] = cooldowns
        save_model_state(state)
        return False

    return True


def mark_model_cooldown(model, seconds=None):
    cooldown_seconds = seconds or MODEL_COOLDOWN_SECONDS
    cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)

    state = load_model_state()
    cooldowns = state.get("cooldowns", {})
    cooldowns[model] = cooldown_until.isoformat()
    state["cooldowns"] = cooldowns

    save_model_state(state)

    print(f"Model cooldown'a alındı: {model} - {cooldown_seconds} saniye")


def get_retry_after_seconds(response):
    retry_after = response.headers.get("Retry-After")

    if not retry_after:
        return None

    try:
        return int(float(retry_after))
    except Exception:
        return None


def get_available_models(model_pool):
    available = []

    for model in model_pool:
        if not is_model_on_cooldown(model):
            available.append(model)

    if available:
        return available

    return model_pool


def get_headers():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY bulunamadı. .env veya Render Environment Variables kontrol edilmeli.")

    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://telegram-ai-daily-bot.onrender.com",
        "X-Title": "Telegram AI Daily Bot"
    }


def is_invalid_answer(answer):
    if answer is None:
        return True

    if not isinstance(answer, str):
        return True

    cleaned = answer.strip()
    cleaned_lower = cleaned.lower()

    if cleaned in ["", "[]", "{}", "null", "None", "none", "undefined"]:
        return True

    bad_starts = [
        "user safety:",
        "safe",
        "unsafe"
    ]

    for item in bad_starts:
        if cleaned_lower.startswith(item):
            return True

    return False


def extract_message_from_response(response_json):
    choices = response_json.get("choices", [])

    if not choices:
        raise ValueError("OpenRouter response içinde choices bulunamadı.")

    message = choices[0].get("message", {})

    if not message:
        raise ValueError("OpenRouter response içinde message bulunamadı.")

    return message


def request_with_model_failover(payload, model_pool):
    if not model_pool:
        raise ValueError("Model pool boş. .env içindeki model değerlerini kontrol et.")

    models_to_try = get_available_models(model_pool)
    last_error = None

    for model in models_to_try:
        payload_with_model = dict(payload)
        payload_with_model["model"] = model

        try:
            print(f"OpenRouter isteği gönderiliyor. Model: {model}")

            response = requests.post(
                OPENROUTER_API_URL,
                headers=get_headers(),
                json=payload_with_model,
                timeout=90
            )

            if response.status_code in TEMPORARY_ERROR_STATUS_CODES:
                retry_after = get_retry_after_seconds(response)
                mark_model_cooldown(
                    model=model,
                    seconds=retry_after or MODEL_COOLDOWN_SECONDS
                )

                last_error = f"{response.status_code} - {response.text}"
                print(f"Geçici model hatası: {last_error}")
                continue

            if response.status_code in [401, 403]:
                raise RuntimeError(
                    f"OpenRouter yetki hatası: {response.status_code} - {response.text}"
                )

            if response.status_code >= 400:
                raise RuntimeError(
                    f"OpenRouter hata döndürdü: {response.status_code} - {response.text}"
                )

            response_json = response.json()
            return response_json

        except Exception as error:
            last_error = str(error)
            print(f"Model isteği başarısız: {model} - {error}")

            mark_model_cooldown(
                model=model,
                seconds=MODEL_COOLDOWN_SECONDS
            )

    raise RuntimeError(f"Tüm modeller başarısız oldu. Son hata: {last_error}")


def send_chat_request(
    messages,
    temperature=0.3,
    max_tokens=700,
    model_pool=None,
    tools=None,
    tool_choice=None,
    response_format=None,
    return_message=False
):
    selected_pool = model_pool or CHAT_MODEL_POOL

    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if tools is not None:
        payload["tools"] = tools

    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    if response_format is not None:
        payload["response_format"] = response_format

    response_json = request_with_model_failover(
        payload=payload,
        model_pool=selected_pool
    )

    message = extract_message_from_response(response_json)

    if return_message:
        return message

    content = message.get("content", "")

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        content = "\n".join(text_parts)

    if is_invalid_answer(content):
        raise RuntimeError("OpenRouter geçersiz/boş cevap döndürdü.")

    return content.strip()


def ask_openrouter(user_text, history_context=""):
    system_prompt = """
You are a helpful daily-life assistant.

You can help with:
- daily planning
- focus and productivity
- routines
- breaks
- simple meal ideas
- weather-based preparation

Keep answers short, practical and safe.
If the question is about medical, legal or financial advice, do not give a direct professional decision.
Give a safe general response and recommend consulting a qualified professional when needed.
"""

    user_prompt = f"""
Conversation history:
{history_context}

User message:
{user_text}
"""

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.4,
            max_tokens=600,
            model_pool=CHAT_MODEL_POOL
        )

    except Exception as error:
        print(f"ask_openrouter error: {error}")
        return "Şu an yapay zeka modeli düzgün bir cevap üretemedi. Lütfen birkaç saniye sonra tekrar dener misin?"


def ask_openrouter_with_context(user_text, context_text, history_context=""):
    system_prompt = """
You are a helpful assistant for a daily-life Telegram and web bot.

Use the provided context or service result to answer the user.
If the user writes in Turkish, answer in Turkish.
If the user writes in English, answer in English.
Keep the answer short, clear and practical.
Do not mention internal technical details.
"""

    user_prompt = f"""
Conversation history:
{history_context}

Context / service result:
{context_text}

User message:
{user_text}

Now answer the user using the context.
"""

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3,
            max_tokens=700,
            model_pool=CHAT_MODEL_POOL
        )

    except Exception as error:
        print(f"ask_openrouter_with_context error: {error}")
        return "Şu an yapay zeka modeli düzgün bir cevap üretemedi. Lütfen birkaç saniye sonra tekrar dener misin?"


def ask_openrouter_with_tool_result(
    user_text,
    assistant_message,
    tool_call,
    tool_result_text,
    history_context=""
):
    system_prompt = """
You are a helpful assistant for a daily-life Telegram and web bot.

The tool result contains real service data.
Use the tool result to answer the user.
If the user asks about weather, convert raw weather data into practical daily advice.
If the user writes in Turkish, answer in Turkish.
If the user writes in English, answer in English.
Keep the answer concise and useful.
Do not mention internal technical details.
"""

    user_prompt = f"""
Conversation history:
{history_context}

User message:
{user_text}

Tool call:
{json.dumps(tool_call, ensure_ascii=False) if tool_call else ""}

Tool result:
{tool_result_text}

Now answer the user based on the tool result.
"""

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.3,
            max_tokens=700,
            model_pool=CHAT_MODEL_POOL
        )

    except Exception as error:
        print(f"ask_openrouter_with_tool_result error: {error}")
        return "Hava durumu verisini aldım fakat şu an cevabı oluştururken sorun yaşadım. Lütfen tekrar dener misin?"


def get_weather_tool_schema():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city name, for example Izmir, Istanbul, Ankara, Paris."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]


def get_tool_call_decision(user_text, history_context=""):
    system_prompt = """
You are a tool router for a Telegram and web assistant.

Decide whether the user needs a tool call.

Available tool:
- get_weather(city): use only for weather, rain, temperature, umbrella, jacket, wind or outside preparation questions.

Rules:
- If the user asks weather-related questions and includes a city, call get_weather.
- If the user asks weather-related questions but no city is clear, do not call a tool.
- For daily planning, routines, focus, breaks or general chat, do not call a tool.
- For unclear messages, do not call a tool.
"""

    user_prompt = f"""
Conversation history:
{history_context}

User message:
{user_text}
"""

    try:
        message = send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0,
            max_tokens=300,
            model_pool=TOOL_MODEL_POOL,
            tools=get_weather_tool_schema(),
            tool_choice="auto",
            return_message=True
        )

        return {
            "assistant_message": message.get("content", ""),
            "tool_calls": message.get("tool_calls", []),
            "raw_message": message
        }

    except Exception as error:
        print(f"get_tool_call_decision error: {error}")

        return {
            "assistant_message": "",
            "tool_calls": [],
            "raw_message": None,
            "error": str(error)
        }


def extract_json_from_text(text):
    if not text:
        return None

    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        return None

    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return None


def get_json_tool_decision(user_text, history_context=""):
    system_prompt = """
You are a strict JSON tool router.

Return only valid JSON in this format:
{
  "tool": "weather" or "none",
  "city": "city name or empty string",
  "reason": "short reason"
}

Use weather only for weather, rain, umbrella, temperature, jacket, wind or outside preparation questions.
"""

    user_prompt = f"""
Conversation history:
{history_context}

User message:
{user_text}
"""

    try:
        answer = send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0,
            max_tokens=250,
            model_pool=TOOL_MODEL_POOL
        )

        parsed = extract_json_from_text(answer)

        if not parsed:
            return {
                "tool": "none",
                "city": "",
                "reason": "JSON parse failed"
            }

        return parsed

    except Exception as error:
        print(f"get_json_tool_decision error: {error}")

        return {
            "tool": "none",
            "city": "",
            "reason": str(error)
        }


def summarize_session(transcript):
    system_prompt = """
You summarize a conversation session for a daily-life assistant.

Write a short summary in Turkish.
Include:
- user's main needs
- important preferences
- unresolved topics
Keep it under 120 words.
"""

    user_prompt = f"""
Conversation transcript:
{transcript}
"""

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_tokens=300,
            model_pool=CHAT_MODEL_POOL
        )

    except Exception as error:
        print(f"summarize_session error: {error}")
        return ""


def guess_mime_type(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if mime_type:
        return mime_type

    extension = os.path.splitext(image_path)[1].lower()

    if extension in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if extension == ".png":
        return "image/png"

    if extension == ".webp":
        return "image/webp"

    return "image/jpeg"


def encode_file_base64(file_path):
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def analyze_image_with_openrouter(image_path, caption="", history_context=""):
    image_base64 = encode_file_base64(image_path)
    mime_type = guess_mime_type(image_path)

    system_prompt = """
You are a helpful visual assistant for a daily-life Telegram and web bot.

Analyze the image safely and practically.
If the user asks about a workspace, routine, simple food or daily organization, give helpful suggestions.
If the image is unrelated, briefly describe what you can see.
If the user writes in Turkish, answer in Turkish.
If the user writes in English, answer in English.
Do not identify real people.
"""

    user_text = caption or "Bu görseli kısaca açıklar mısın?"

    content = [
        {
            "type": "text",
            "text": f"""
Conversation history:
{history_context}

User caption:
{user_text}
"""
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_base64}"
            }
        }
    ]

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            temperature=0.3,
            max_tokens=700,
            model_pool=VISION_MODEL_POOL
        )

    except Exception as error:
        print(f"analyze_image_with_openrouter error: {error}")
        return "Fotoğrafı analiz ederken bir sorun oluştu. Lütfen tekrar dener misin?"

def ask_openrouter_with_rag(user_text, rag_context, history_context=""):
    from rag_utils import detect_user_language

    detected_language = detect_user_language(user_text)

    if detected_language == "en":
        language_instruction = (
            "The user's message is in English. Answer in English. "
            "You may use Turkish knowledge base content, but explain it naturally in English."
        )
        fallback_message = (
            "The AI model could not generate a proper answer right now. "
            "Please try again in a few seconds."
        )
    else:
        language_instruction = (
            "Kullanıcının mesajı Türkçe. Cevabı Türkçe ver."
        )
        fallback_message = (
            "Şu an yapay zeka modeli düzgün bir cevap üretemedi. "
            "Lütfen birkaç saniye sonra tekrar dener misin?"
        )

    system_prompt = f"""
You are a helpful assistant for a daily-life Telegram and web bot.

You must follow these rules:
- Use only the provided knowledge base chunks when answering knowledge-based questions.
- Do not copy the chunks directly. Synthesize the answer in your own words.
- The knowledge base may include multiple chunks. Choose the most relevant chunks yourself.
- If the chunks are not enough to answer, clearly say that the knowledge base does not contain enough information.
- Do not invent facts that are not supported by the provided chunks.
- Use the conversation history only to understand follow-up messages such as "continue", "according to that", "az önce", "ona göre".
- Keep the answer practical, short and useful.
- Do not mention internal technical details like RAG, threshold, chunks, scores or embeddings.
- Do not include page numbers, page headers, chunk IDs, topic labels or keyword lists in the final answer.
- Add only the source file name at the end.
- {language_instruction}
"""

    user_prompt = f"""
Conversation history:
{history_context}

Retrieved knowledge base content:
{rag_context}

User message:
{user_text}

Now answer the user according to the rules. Do not copy the source text directly.
"""

    try:
        answer = send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.2,
            max_tokens=700,
            model_pool=CHAT_MODEL_POOL
        )

        return answer

    except Exception as error:
        print(f"RAG OpenRouter error: {error}")
        return fallback_message


def ask_openrouter_general_fallback(user_text, history_context=""):
    from rag_utils import detect_user_language

    detected_language = detect_user_language(user_text)

    if detected_language == "en":
        language_instruction = "Answer in English."
    else:
        language_instruction = "Cevabı Türkçe ver."

    system_prompt = f"""
You are a limited daily-life assistant.

You may answer basic safe general questions, but you should not act as a professional advisor.

Rules:
- For finance, investment, medical, legal or high-risk topics, do not give direct advice.
- For meaningless repeated input, ask the user to write more clearly.
- If the question is outside your main scope, answer briefly and redirect to your scope.
- Main scope: daily planning, focus, breaks, routines, simple meal ideas, weather-based preparation.
- {language_instruction}
"""

    user_prompt = f"""
Conversation history:
{history_context}

User message:
{user_text}
"""

    try:
        return send_chat_request(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            temperature=0.4,
            max_tokens=500,
            model_pool=CHAT_MODEL_POOL
        )

    except Exception as error:
        print(f"ask_openrouter_general_fallback error: {error}")

        if detected_language == "en":
            return (
                "I could not answer that properly right now. "
                "I can mainly help with daily planning, focus, routines, breaks, simple meal ideas and weather preparation."
            )

        return (
            "Şu an buna düzgün cevap veremedim. "
            "Ben daha çok günlük planlama, odaklanma, rutin, mola yönetimi, basit yemek fikirleri ve hava durumuna göre hazırlık konularında yardımcı olabilirim."
        )