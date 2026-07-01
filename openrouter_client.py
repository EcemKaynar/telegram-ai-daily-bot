import base64
import json
import mimetypes
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv(".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-4-26b-a4b-it:free")
OPENROUTER_TOOL_MODEL = os.getenv("OPENROUTER_TOOL_MODEL", "openrouter/free")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_STATE_FILE = os.getenv("MODEL_STATE_FILE", "model_state.json")
MODEL_COOLDOWN_SECONDS = int(os.getenv("MODEL_COOLDOWN_SECONDS", "900"))

TRANSIENT_STATUS_CODES = [429, 502, 503, 529]


WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": (
            "Get live current weather information for a city or location. "
            "Use this when the user asks about weather, temperature, rain, wind, umbrella, "
            "what to wear according to weather, or daily plans that require live weather data. "
            "If the user gave a city earlier in the conversation, use that city."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City or location name mentioned by the user or previous conversation context."
                }
            },
            "required": ["city"]
        }
    }
}


def parse_model_pool(env_name, default_models):
    value = os.getenv(env_name)

    if value:
        models = [model.strip() for model in value.split(",") if model.strip()]
    else:
        models = default_models

    unique_models = []

    for model in models:
        if model and model not in unique_models:
            unique_models.append(model)

    return unique_models


CHAT_MODEL_POOL = parse_model_pool(
    "OPENROUTER_CHAT_MODEL_POOL",
    [OPENROUTER_MODEL, "openrouter/free"]
)

TOOL_MODEL_POOL = parse_model_pool(
    "OPENROUTER_TOOL_MODEL_POOL",
    [OPENROUTER_TOOL_MODEL, "openrouter/free"]
)

VISION_MODEL_POOL = parse_model_pool(
    "OPENROUTER_VISION_MODEL_POOL",
    ["openrouter/free"]
)


def load_model_state():
    if not os.path.exists(MODEL_STATE_FILE):
        return {}

    try:
        with open(MODEL_STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_model_state(state):
    with open(MODEL_STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(state, file, ensure_ascii=False, indent=2)


def get_pool_state(state, pool_name):
    if pool_name not in state:
        state[pool_name] = {
            "active_model": None,
            "cooldowns": {},
            "last_errors": {}
        }

    return state[pool_name]


def cleanup_expired_cooldowns(pool_state):
    now = time.time()
    cooldowns = pool_state.get("cooldowns", {})

    expired_models = []

    for model, available_at in cooldowns.items():
        if now >= available_at:
            expired_models.append(model)

    for model in expired_models:
        del cooldowns[model]


def get_available_models(pool_name, model_pool):
    state = load_model_state()
    pool_state = get_pool_state(state, pool_name)

    cleanup_expired_cooldowns(pool_state)

    active_model = pool_state.get("active_model")

    if active_model not in model_pool:
        active_model = model_pool[0] if model_pool else None
        pool_state["active_model"] = active_model

    ordered_models = []

    if active_model:
        ordered_models.append(active_model)

    for model in model_pool:
        if model not in ordered_models:
            ordered_models.append(model)

    now = time.time()
    available_models = []

    for model in ordered_models:
        cooldown_until = pool_state.get("cooldowns", {}).get(model)

        if not cooldown_until or now >= cooldown_until:
            available_models.append(model)

    save_model_state(state)

    return available_models


def mark_model_success(pool_name, model):
    state = load_model_state()
    pool_state = get_pool_state(state, pool_name)

    pool_state["active_model"] = model

    if model in pool_state.get("cooldowns", {}):
        del pool_state["cooldowns"][model]

    pool_state["last_success_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    save_model_state(state)


def mark_model_limited(pool_name, model, status_code, retry_after_seconds=None, error_text=""):
    state = load_model_state()
    pool_state = get_pool_state(state, pool_name)

    cooldown_seconds = retry_after_seconds or MODEL_COOLDOWN_SECONDS
    available_at = time.time() + cooldown_seconds

    pool_state.setdefault("cooldowns", {})[model] = available_at
    pool_state.setdefault("last_errors", {})[model] = {
        "status_code": status_code,
        "error_text": error_text[:300],
        "cooldown_seconds": cooldown_seconds,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if pool_state.get("active_model") == model:
        pool_state["active_model"] = None

    save_model_state(state)

    print(
        f"{model} limit/provider hatası aldı. "
        f"Kod: {status_code}. "
        f"{cooldown_seconds} saniye cooldown'a alındı."
    )


def parse_retry_after(response):
    retry_after = response.headers.get("Retry-After")

    if not retry_after:
        return None

    try:
        return int(retry_after)
    except Exception:
        return None


def get_headers():
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY bulunamadı. .env dosyasını kontrol et.")

    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Telegram AI Bot"
    }


def request_with_model_failover(
    pool_name,
    model_pool,
    messages,
    temperature=0.5,
    max_tokens=700,
    tools=None,
    tool_choice=None
):
    available_models = get_available_models(pool_name, model_pool)

    if not available_models:
        return None, "Tüm yedek modeller geçici olarak limitte veya cooldown durumunda."

    last_error = None

    for model in available_models:
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            data["tools"] = tools

        if tool_choice:
            data["tool_choice"] = tool_choice

        try:
            response = requests.post(
                OPENROUTER_URL,
                headers=get_headers(),
                json=data,
                timeout=60
            )

            if response.status_code in TRANSIENT_STATUS_CODES:
                retry_after_seconds = parse_retry_after(response)

                mark_model_limited(
                    pool_name=pool_name,
                    model=model,
                    status_code=response.status_code,
                    retry_after_seconds=retry_after_seconds,
                    error_text=response.text
                )

                last_error = f"{model} geçici hata verdi: {response.status_code}"
                continue

            if response.status_code == 404:
                mark_model_limited(
                    pool_name=pool_name,
                    model=model,
                    status_code=response.status_code,
                    retry_after_seconds=86400,
                    error_text=response.text
                )

                last_error = f"{model} bulunamadı veya erişilemiyor."
                continue

            if response.status_code in [401, 402, 403]:
                return None, f"OpenRouter yetki/ödeme hatası: {response.status_code} - {response.text[:300]}"

            response.raise_for_status()

            result = response.json()
            mark_model_success(pool_name, model)

            print(f"Aktif kullanılan model ({pool_name}): {model}")
            return result, None

        except requests.exceptions.HTTPError as error:
            last_error = f"{model} HTTP hatası: {error}"
            print(last_error)
            continue

        except Exception as error:
            last_error = f"{model} genel hata: {error}"
            print(last_error)
            continue

    return None, last_error or "Uygun model bulunamadı."


def is_invalid_answer(answer):
    if answer is None:
        return True

    if not isinstance(answer, str):
        return True

    cleaned_answer = answer.strip()

    invalid_values = [
        "",
        "[]",
        "{}",
        "null",
        "None",
        "none",
        "undefined",
        "safe"
    ]

    if cleaned_answer in invalid_values:
        return True

    if cleaned_answer.lower() == "safe":
        return True

    if cleaned_answer.startswith("User Safety:"):
        return True

    if cleaned_answer.startswith("Safety:"):
        return True

    return False


def send_chat_request(messages, pool_name="chat", model_pool=None, temperature=0.5, max_tokens=700):
    if model_pool is None:
        model_pool = CHAT_MODEL_POOL

    result, error = request_with_model_failover(
        pool_name=pool_name,
        model_pool=model_pool,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if not result:
        print(f"Model havuzu cevap veremedi: {error}")
        return (
            "Şu an ücretsiz yapay zeka modelleri yoğunluk nedeniyle cevap veremiyor. "
            "Lütfen birkaç dakika sonra tekrar deneyin."
        )

    message = result["choices"][0]["message"]
    answer = (message.get("content") or "").strip()

    if is_invalid_answer(answer):
        print(f"Model geçersiz cevap döndürdü: {answer}")
        return (
            "Şu an yapay zeka modeli düzgün bir cevap üretemedi. "
            "Lütfen birkaç saniye sonra tekrar dener misin?"
        )

    return answer


def ask_openrouter(user_text, history_context=""):
    system_prompt = (
        "Sen günlük yaşamı kolaylaştıran yardımcı bir Telegram botusun. "
        "Kapsamın: günlük planlama, kişisel verimlilik, motivasyon, rutin oluşturma, "
        "çalışma düzeni, plan defteri/to-do listesi, mola yönetimi, kahve/çay gibi günlük içecekler, "
        "kısa dinlenme, odaklanma, basit yemek önerileri, çalışma ortamı düzeni, "
        "kıyafet ve hava durumuna göre günlük hazırlık önerileridir. "
        "Kullanıcı bu konularla ilgili soru sorarsa kısa, net, samimi ve uygulanabilir cevap ver. "
        "Kullanıcı Türkçe yazarsa doğal ve anlaşılır Türkçe cevap ver. "
        "Kullanıcı başka bir dilde yazarsa aynı dilde cevap ver. "
        "Geçmiş konuşma verildiyse bunu sadece bağlamı anlamak ve aynı soruları tekrar sormamak için kullan. "
        "Konu dışı sorulara cevap verme. "
        "Eğer kullanıcı konu dışında bir şey sorarsa kibarca şu kapsamda yardımcı olabileceğini söyle: "
        "günlük planlama, verimlilik, motivasyon, rutin, mola yönetimi, çalışma ortamı, plan defteri, "
        "kahve/çay gibi günlük rutinler, basit yemek önerileri ve hava durumu. "
        "Tıbbi teşhis koyma, diyetisyen gibi kesin beslenme planı verme, hukuki/finansal danışmanlık verme, siyasi yorum yapma, kod yazma. "
        "Sağlık veya beslenme konusunda cevap verirken genel ve güvenli öneriler ver; hastalık/tedavi iddiasında bulunma. "
        "Markdown kullanma. Yıldız, başlık işareti veya özel formatlama kullanma. "
        "Madde işareti kullanacaksan sadece normal tire (-) kullan. "
        "En fazla 5 kısa maddeyle cevap ver."
    )

    user_prompt = user_text

    if history_context:
        user_prompt = (
            f"Konuşma geçmişi:\n{history_context}\n\n"
            f"Kullanıcının yeni mesajı:\n{user_text}\n\n"
            "Cevabını yeni mesaja ver. Geçmişi sadece bağlam için kullan."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return send_chat_request(
        messages=messages,
        pool_name="chat",
        model_pool=CHAT_MODEL_POOL,
        temperature=0.5,
        max_tokens=700
    )


def ask_openrouter_with_context(user_text, context_text, history_context=""):
    system_prompt = (
        "Sen günlük yaşamı kolaylaştıran yardımcı bir Telegram botusun. "
        "Kapsamın: günlük planlama, kişisel verimlilik, motivasyon, rutin oluşturma, "
        "çalışma düzeni, plan defteri/to-do listesi, mola yönetimi, kahve/çay gibi günlük içecekler, "
        "kısa dinlenme, odaklanma, basit yemek önerileri, çalışma ortamı düzeni, "
        "kıyafet ve hava durumuna göre günlük hazırlık önerileridir. "
        "Sana canlı servis verisi verilmişse cevabını sadece bu servis verisine göre üret. "
        "Servis verisi varken kendi hafızandan hava durumu uydurma. "
        "Kullanıcı hava durumuna göre öneri istiyorsa sıcaklık, yağış, rüzgar ve hissedilen sıcaklığı dikkate al. "
        "Kullanıcı Türkçe yazarsa doğal ve anlaşılır Türkçe cevap ver. "
        "Kullanıcı başka bir dilde yazarsa aynı dilde cevap ver. "
        "Geçmiş konuşma varsa bunu sadece bağlamı anlamak için kullan. "
        "Sağlık, kahve, yemek veya beslenme konularında genel ve güvenli öneriler ver; tıbbi teşhis veya kesin diyet planı verme. "
        "Markdown kullanma. Yıldız veya özel formatlama kullanma. "
        "Madde işareti kullanacaksan sadece normal tire (-) kullan. "
        "Cevabın kısa, net ve pratik olsun."
    )

    user_prompt = (
        f"Kullanıcı mesajı: {user_text}\n\n"
        f"Canlı servis verisi:\n{context_text}\n\n"
        "Bu canlı servis verisine göre kullanıcıya cevap ver."
    )

    if history_context:
        user_prompt = (
            f"Konuşma geçmişi:\n{history_context}\n\n"
            f"{user_prompt}"
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return send_chat_request(
        messages=messages,
        pool_name="chat",
        model_pool=CHAT_MODEL_POOL,
        temperature=0.4,
        max_tokens=700
    )


def sanitize_assistant_tool_message(assistant_message):
    if not assistant_message:
        return {
            "role": "assistant",
            "content": None
        }

    sanitized = {
        "role": "assistant",
        "content": assistant_message.get("content"),
    }

    if assistant_message.get("tool_calls"):
        sanitized["tool_calls"] = assistant_message.get("tool_calls")

    return sanitized


def ask_openrouter_with_tool_result(user_text, assistant_message, tool_call, tool_result_text, history_context=""):
    if not assistant_message or not tool_call:
        return ask_openrouter_with_context(user_text, tool_result_text, history_context)

    system_prompt = (
        "Sen günlük yaşamı kolaylaştıran yardımcı bir Telegram botusun. "
        "Kapsamın: günlük planlama, kişisel verimlilik, motivasyon, rutin oluşturma, "
        "çalışma düzeni, plan defteri/to-do listesi, mola yönetimi, kahve/çay gibi günlük içecekler, "
        "kısa dinlenme, odaklanma, basit yemek önerileri, çalışma ortamı düzeni, "
        "kıyafet ve hava durumuna göre günlük hazırlık önerileridir. "
        "Sana tool sonucuyla canlı hava durumu verisi verilmişse cevabını sadece bu veriye göre üret. "
        "Servis verisi varken kendi hafızandan hava durumu uydurma. "
        "Kullanıcı hava durumuna göre kıyafet, şemsiye, yürüyüş, dışarı çıkma gibi öneriler istiyorsa canlı hava verisini dikkate al. "
        "Kullanıcı Türkçe yazarsa doğal ve anlaşılır Türkçe cevap ver. "
        "Kullanıcı başka bir dilde yazarsa aynı dilde cevap ver. "
        "Geçmiş konuşma varsa bunu sadece bağlamı anlamak için kullan. "
        "Sağlık, kahve, yemek veya beslenme konularında genel ve güvenli öneriler ver; tıbbi teşhis veya kesin diyet planı verme. "
        "Markdown kullanma. Yıldız veya özel formatlama kullanma. "
        "Madde işareti kullanacaksan sadece normal tire (-) kullan. "
        "Cevabın kısa, net ve pratik olsun."
    )

    tool_call_id = tool_call.get("id")
    tool_name = tool_call.get("function", {}).get("name", "get_current_weather")

    if not tool_call_id:
        return ask_openrouter_with_context(user_text, tool_result_text, history_context)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    if history_context:
        messages.append(
            {
                "role": "user",
                "content": f"Konuşma geçmişi:\n{history_context}"
            }
        )

    messages.extend(
        [
            {
                "role": "user",
                "content": user_text
            },
            sanitize_assistant_tool_message(assistant_message),
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": tool_result_text
            }
        ]
    )

    return send_chat_request(
        messages=messages,
        pool_name="chat",
        model_pool=CHAT_MODEL_POOL,
        temperature=0.4,
        max_tokens=700
    )


def extract_json_from_text(text):
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None

    return None


def get_json_tool_decision(user_text):
    system_prompt = (
        "You are a tool router. "
        "Analyze the user's message semantically. "
        "Return only valid JSON. Do not write explanation. "
        "If the user asks about live weather, temperature, rain, wind, umbrella, what to wear based on weather, "
        "or daily plans that require live weather, return: "
        '{"tool": "weather", "city": "city name"} '
        "If no live weather data is needed, return: "
        '{"tool": "none", "city": null} '
        "Extract the city from the user's message or previous conversation context. "
        "Keep the city in its original form if possible."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    answer = send_chat_request(
        messages=messages,
        pool_name="tool",
        model_pool=TOOL_MODEL_POOL,
        temperature=0,
        max_tokens=300
    )

    parsed = extract_json_from_text(answer)

    return {
        "role": "assistant",
        "content": answer,
        "tool_calls": [],
        "json_decision": parsed
    }


def get_tool_call_decision(user_text):
    system_prompt = (
        "You are a tool-calling router for a Telegram assistant. "
        "Decide whether the user's message requires live weather data. "
        "If live weather data is needed, call the get_current_weather tool with the city parameter. "
        "If the city was mentioned earlier in the previous conversation context, use that city. "
        "If live weather data is not needed, do not call any tool. "
        "Do not answer the user directly."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_text
        }
    ]

    result, error = request_with_model_failover(
        pool_name="tool",
        model_pool=TOOL_MODEL_POOL,
        messages=messages,
        temperature=0,
        max_tokens=300,
        tools=[WEATHER_TOOL_SCHEMA],
        tool_choice="auto"
    )

    if not result:
        print(f"Tool calling model havuzu hata verdi, JSON fallback deneniyor: {error}")
        return get_json_tool_decision(user_text)

    message = result["choices"][0]["message"]

    print("Tool decision message:", message)

    if message.get("tool_calls"):
        return message

    print("Tool call dönmedi. JSON router fallback deneniyor.")
    return get_json_tool_decision(user_text)


def summarize_session(previous_summary, transcript_text):
    system_prompt = (
        "Sen bir konuşma özeti çıkarma aracısın. "
        "Kullanıcının tercihlerini, yaşadığı şehir bilgisini, önceki sorularını ve önemli bağlamı kısa şekilde özetle. "
        "Gereksiz detay yazma. "
        "Özet, botun sonraki mesajlarda kullanıcıya aynı soruları tekrar sormamasına yardımcı olmalı. "
        "Türkçe özet yaz."
    )

    user_prompt = (
        f"Önceki özet:\n{previous_summary}\n\n"
        f"Konuşma dökümü:\n{transcript_text}\n\n"
        "Güncel kısa oturum özetini yaz."
    )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    return send_chat_request(
        messages=messages,
        pool_name="chat",
        model_pool=CHAT_MODEL_POOL,
        temperature=0.2,
        max_tokens=500
    )


def encode_image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)

    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded_image}"


def analyze_image_with_openrouter(image_path, caption="", history_context=""):
    image_data_url = encode_image_to_data_url(image_path)

    system_prompt = (
        "Sen bir Telegram botu için fotoğraf analiz eden vision asistansın. "
        "Botun konusu günlük yaşamı kolaylaştırmak üzerinedir. "
        "Kapsamın: günlük planlama, kişisel verimlilik, motivasyon, rutin oluşturma, "
        "çalışma alanı düzeni, plan defteri, yapılacaklar listesi, takvim, kahve/çay gibi günlük içecekler, "
        "kahve molası, kısa dinlenme, odaklanma, basit yemekler, öğün hazırlığı, market/alışveriş hazırlığı, "
        "uyku ve dinlenme rutini, hafif egzersiz, dış hava/gökyüzü, kıyafet, çanta, şemsiye ve hava durumuna göre günlük hazırlıktır. "
        "Fotoğraf bu konularla ilgiliyse fotoğrafta gördüklerini kısa şekilde yorumla ve kullanıcıya pratik öneriler ver. "
        "Plan defteri, ajanda, takvim veya yapılacaklar listesi görürsen planlama yaptığını anlayıp görevleri önceliklendirme önerisi ver. "
        "Kahve veya içecek görürsen bunu günlük mola, odaklanma, kısa dinlenme veya çalışma rutini bağlamında yorumla. "
        "Kullanıcı özellikle sormadıkça sağlık, kafein veya su tüketimi hakkında uyarı verme. "
        "Yemek görürsen dengeli ve pratik bir öğün olup olmadığı hakkında genel yorum yap; kesin kalori, teşhis veya diyet planı verme. "
        "Masa veya çalışma ortamı görürsen odaklanma ve düzen önerisi ver. "
        "Gökyüzü/dış hava görürsen görselden genel yorum yap; kesin hava durumu için şehir bilgisi gerekebileceğini belirt. "
        "Fotoğraf konu dışıysa kibarca bu kapsamda yardımcı olamayacağını söyle. "
        "Kullanıcı Türkçe yazarsa Türkçe cevap ver. Kullanıcı başka dilde yazarsa aynı dilde cevap ver. "
        "Kesin emin olmadığın şeyleri kesinmiş gibi söyleme. "
        "Fotoğraftaki kişiler hakkında hassas veya kişisel çıkarım yapma. "
        "Markdown kullanma. Cevabın kısa, net, tatlı ve anlaşılır olsun."
    )

    user_text = caption if caption else "Bu fotoğrafı analiz eder misin?"

    if history_context:
        user_text = (
            f"Konuşma geçmişi:\n{history_context}\n\n"
            f"Kullanıcının fotoğrafla birlikte yazdığı mesaj:\n{user_text}"
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            ]
        }
    ]

    result, error = request_with_model_failover(
        pool_name="vision",
        model_pool=VISION_MODEL_POOL,
        messages=messages,
        temperature=0.4,
        max_tokens=700
    )

    if not result:
        print(f"Vision model havuzu cevap veremedi: {error}")
        return (
            "Şu an görüntü analiz modeli yoğunluk nedeniyle cevap veremiyor. "
            "Lütfen birkaç dakika sonra tekrar deneyin."
        )

    message = result["choices"][0]["message"]
    answer = (message.get("content") or "").strip()

    if is_invalid_answer(answer):
        return "Şu an fotoğraf için geçerli bir analiz üretemedim. Lütfen tekrar dener misin?"

    return answer

def ask_openrouter_with_rag(user_text, rag_context, history_context=""):
    from rag_utils import detect_user_language

    detected_language = detect_user_language(user_text)

    if detected_language == "en":
        language_instruction = (
            "The user's message is in English. Answer in English. "
            "You may use the Turkish knowledge base content, but translate and explain it naturally in English."
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
- Answer only by using the provided knowledge base content.
- Do not invent information that is not supported by the knowledge base.
- If the knowledge base is not enough, say that the knowledge base does not contain enough information.
- Keep the answer clear, short and useful.
- Do not mention internal technical details like RAG, embeddings or model errors.
- {language_instruction}
"""

    user_prompt = f"""
Conversation history:
{history_context}

Knowledge base content:
{rag_context}

User message:
{user_text}

Now answer the user according to the rules.
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
            max_tokens=600
        )

        return answer

    except Exception as error:
        print(f"RAG OpenRouter error: {error}")
        return fallback_message