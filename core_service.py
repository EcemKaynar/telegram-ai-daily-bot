import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from openrouter_client import (
    ask_openrouter_with_context,
    ask_openrouter_with_tool_result,
    analyze_image_with_openrouter,
    ask_openrouter_with_rag,
    ask_openrouter_general_fallback
)

from tool_router import detect_tool_call
from tools.weather_tool import get_weather_by_city, to_json_text
from database import save_interaction, save_service_log, save_rag_log
from history_manager import get_session_and_history
from voice_utils import speech_to_text, text_to_speech
from rag_utils import (
    search_knowledge_base,
    format_rag_context,
    get_rag_sources_json,
    build_direct_rag_answer,
    detect_user_language
)


def now_ms():
    return time.perf_counter()


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def is_bad_ai_answer(ai_answer):
    if ai_answer is None:
        return True

    if not isinstance(ai_answer, str):
        return True

    cleaned = ai_answer.strip()
    cleaned_lower = cleaned.lower()

    if cleaned in ["", "[]", "{}", "null", "None", "none", "undefined"]:
        return True

    bad_phrases = [
        "şu an yapay zeka modeli düzgün bir cevap üretemedi",
        "yapay zeka modeli düzgün bir cevap üretemedi",
        "birkaç saniye sonra tekrar dener misin",
        "ai model could not generate",
        "model could not generate",
        "i could not generate a proper answer",
        "please try again in a few seconds"
        "<pad>",
    ]

    return any(phrase in cleaned_lower for phrase in bad_phrases)
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

    # Bazı modeller markdown bold işaretlerini fazla basıyor.
    answer = answer.replace("**", "")

    return answer.strip()

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


def normalize_basic_text(user_text):
    text = str(user_text or "").lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\wçğıöşü\s]", "", text)
    text = text.strip()

    return text


def is_meaningless_input(user_text):
    text = normalize_basic_text(user_text)
    compact = text.replace(" ", "")

    if not compact:
        return True

    if len(compact) >= 5:
        unique_chars = set(compact)

        if len(unique_chars) <= 2:
            return True

        most_common_count = max(compact.count(char) for char in unique_chars)

        if most_common_count / len(compact) >= 0.75:
            return True

    letter_tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ]+", text)

    if not letter_tokens:
        return True

    return False


def get_meaningless_input_answer(language):
    if language == "en":
        return "I could not understand your message clearly. Could you write it a bit more clearly?"

    return "Mesajını tam anlayamadım. Biraz daha açık şekilde yazar mısın?"


def detect_basic_conversation(user_text):
    text = normalize_basic_text(user_text)
    words = text.split()

    if not text:
        return None

    greetings = [
        "merhaba",
        "selam",
        "selamlar",
        "hello",
        "hi",
        "hey",
        "günaydın",
        "gunaydin",
        "iyi akşamlar",
        "iyi aksamlar",
        "iyi geceler"
    ]

    how_are_you = [
        "nasılsın",
        "nasilsin",
        "naber",
        "ne haber",
        "how are you",
        "how r u",
        "how are u"
    ]

    thanks = [
        "teşekkürler",
        "tesekkurler",
        "teşekkür ederim",
        "tesekkur ederim",
        "sağ ol",
        "sag ol",
        "sağol",
        "sagol",
        "thanks",
        "thank you",
        "thx"
    ]

    help_questions = [
        "ne yapabiliyorsun",
        "neler yapabilirsin",
        "bana nasıl yardımcı olabilirsin",
        "bana nasil yardimci olabilirsin",
        "yardım",
        "yardim",
        "help",
        "what can you do",
        "how can you help me",
        "what do you do"
    ]

    who_are_you = [
        "sen kimsin",
        "kimsin",
        "who are you",
        "what are you"
    ]

    if any(item in text for item in who_are_you):
        return "who_are_you"

    if any(item in text for item in help_questions):
        return "help"

    if any(item in text for item in how_are_you):
        return "how_are_you"

    if text in thanks or any(item in text for item in thanks if len(words) <= 6):
        return "thanks"

    if text in greetings:
        return "greeting"

    if words and words[0] in greetings and len(words) <= 3:
        return "greeting"

    return None


def detect_basic_conversation_language(user_text, default_language):
    text = normalize_basic_text(user_text)

    english_markers = [
        "hello",
        "hi",
        "hey",
        "how are you",
        "thanks",
        "thank you",
        "what can you do",
        "who are you",
        "help"
    ]

    turkish_markers = [
        "merhaba",
        "selam",
        "nasılsın",
        "nasilsin",
        "teşekkür",
        "tesekkur",
        "sağ ol",
        "sag ol",
        "yardım",
        "yardim",
        "sen kimsin"
    ]

    if any(marker in text for marker in english_markers):
        return "en"

    if any(marker in text for marker in turkish_markers):
        return "tr"

    return default_language


def get_basic_conversation_answer(user_text, language):
    intent = detect_basic_conversation(user_text)

    if not intent:
        return None

    language = detect_basic_conversation_language(user_text, language)

    if language == "en":
        if intent == "greeting":
            return (
                "Hello! I can help you with daily planning, focus, breaks, routines, "
                "simple meal ideas and weather-based preparation. How can I help you today?"
            )

        if intent == "how_are_you":
            return (
                "I’m good, thank you! I’m ready to help you plan your day, organize your routine "
                "or improve your focus."
            )

        if intent == "thanks":
            return "You’re welcome!"

        if intent == "help":
            return (
                "I can help with daily planning, productivity, focus, break management, simple routines, "
                "basic meal ideas and weather-based preparation. You can also send voice messages or photos."
            )

        if intent == "who_are_you":
            return (
                "I’m a daily-life assistant bot. I can help with planning, routines, focus, breaks, "
                "simple meal ideas and weather-based preparation."
            )

    else:
        if intent == "greeting":
            return (
                "Merhaba! Günlük planlama, odaklanma, mola yönetimi, rutin oluşturma, "
                "basit yemek fikirleri ve hava durumuna göre hazırlık konularında yardımcı olabilirim. "
                "Bugün sana nasıl yardımcı olayım?"
            )

        if intent == "how_are_you":
            return (
                "İyiyim, teşekkür ederim! Gününü planlama, odaklanma veya rutin oluşturma konusunda "
                "yardımcı olmaya hazırım."
            )

        if intent == "thanks":
            return "Rica ederim!"

        if intent == "help":
            return (
                "Sana günlük planlama, verimlilik, odaklanma, mola yönetimi, basit rutinler, "
                "basit yemek fikirleri ve hava durumuna göre hazırlık konularında yardımcı olabilirim. "
                "İstersen yazılı mesaj, sesli mesaj veya fotoğraf gönderebilirsin."
            )

        if intent == "who_are_you":
            return (
                "Ben günlük yaşamı kolaylaştırmak için hazırlanmış yapay zeka destekli bir asistanım. "
                "Planlama, rutin, odaklanma, mola yönetimi, basit yemek fikirleri ve hava durumu konularında yardımcı olurum."
            )

    return None
def looks_like_weather_question(user_text):
    text = normalize_basic_text(user_text)

    # Net hava durumu kelimeleri
    direct_weather_words = [
        "hava",
        "hava durumu",
        "yağmur",
        "yagmur",
        "yağacak",
        "yagacak",
        "şemsiye",
        "semsiye",
        "sıcaklık",
        "sicaklik",
        "derece",
        "rüzgar",
        "ruzgar",
        "kar",
        "fırtına",
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

    # Kıyafet sorusu hava bağlamında olabilir ama sadece "dışarı" kelimesi yetmez.
    clothing_weather_patterns = [
        "ne giysem",
        "ne giymeliyim",
        "mont gerekir mi",
        "ceket gerekir mi",
        "üşür müyüm",
        "usur muyum",
        "üşürüm",
        "usurum",
        "soğuk mu",
        "soguk mu",
        "sıcak mı",
        "sicak mi"
    ]

    if any(pattern in text for pattern in clothing_weather_patterns):
        return True

    return False

def normalize_location_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_location_lookup_text(text):
    text = str(text or "").lower()

    replacements = {
        "ı": "i",
        "i̇": "i",
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


def clean_extracted_city(city):
    city = normalize_location_text(city)

    city = re.sub(
        r"\b(bugün|bugun|yarın|yarin|şu an|su an|şimdi|simdi|today|tomorrow|now)\b",
        "",
        city,
        flags=re.IGNORECASE
    )

    city = re.sub(
        r"\b(hava|weather|durumu|nasıl|nasil|how|is|the|in|for|için|icin)\b",
        "",
        city,
        flags=re.IGNORECASE
    )

    city = re.sub(r"\s+", " ", city).strip()
    city = city.strip(" '’`.,?!")

    if not city:
        return None

    if len(city) < 2:
        return None

    return city


def extract_city_from_weather_sentence(user_text):
    """
    Burada şehir listesi yok.
    Paris, London, New York, Tokyo, Reykjavik gibi herhangi bir yer adını
    cümle kalıbından çıkarmaya çalışır.
    """

    text = normalize_location_text(user_text)

    patterns = [
        # İstanbul'da hava nasıl, Paris'te hava nasıl
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)['’]?(?:da|de|ta|te)\s+(?:hava|yağmur|sıcaklık|weather)",

        # İstanbul hava durumu, Paris weather
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)\s+(?:hava durumu|weather)",

        # weather in New York, how is the weather in Berlin
        r"(?:weather in|weather for|how is the weather in)\s+(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+)",

        # New York'ta yağmur var mı, Berlin'de yağmur var mı
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)['’]?(?:da|de|ta|te)\s+(?:yağmur|yagmur|rain)",

        # Paris için hava, Berlin için sıcaklık
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)\s+(?:için|icin|for)\s+(?:hava|weather|sıcaklık|sicaklik|temperature)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            city = clean_extracted_city(match.group("city"))

            if city:
                return city

    return None


def history_is_waiting_for_weather_city(history_context):
    text = normalize_location_lookup_text(history_context)

    indicators = [
        "sehir adini yazar misin",
        "hava durumunu kontrol edebilmem icin sehir",
        "i need the city name",
        "need the city name",
        "city name first"
    ]

    return any(indicator in text for indicator in indicators)


def extract_followup_weather_city(user_text, history_context):
    """
    Kullanıcı önce 'hava nasıl?' dedi, bot şehir sordu.
    Sonra kullanıcı sadece 'Paris', 'New York', 'Tokyo' yazarsa bunu şehir kabul eder.
    """

    if not history_is_waiting_for_weather_city(history_context):
        return None

    text = normalize_location_text(user_text)
    words = text.split()

    if 1 <= len(words) <= 5:
        return clean_extracted_city(text)

    return None


def extract_weather_city_with_ai(user_text, history_context):
    """
    Asıl doğru yöntem bu:
    AI/tool router şehir parametresini çıkarır.
    Şehir listesi kullanılmaz.
    """

    tool_call = detect_tool_call(
        user_text=user_text,
        history_context=history_context
    )

    if tool_call.get("tool") == "weather":
        city = tool_call.get("city")

        if city:
            return clean_extracted_city(city), tool_call

    return None, tool_call

def normalize_city_lookup_text(text):
    text = str(text or "").lower()

    replacements = {
        "ı": "i",
        "i̇": "i",
        "ğ": "g",
        "ü": "u",
        "ş": "s",
        "ö": "o",
        "ç": "c"
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def extract_city_from_text(user_text):
    text = normalize_city_lookup_text(user_text)
    tokens = text.split()

    for token in tokens:
        if token in CITY_ALIASES:
            return CITY_ALIASES[token]

        for city_key, city_name in CITY_ALIASES.items():
            if token.startswith(city_key):
                suffix = token[len(city_key):]

                if suffix in CITY_SUFFIXES:
                    return city_name

    return None


def history_is_waiting_for_weather_city(history_context):
    text = normalize_city_lookup_text(history_context)

    indicators = [
        "sehir adini yazar misin",
        "hava durumunu kontrol edebilmem icin sehir",
        "i need the city name",
        "need the city name",
        "city name first"
    ]

    return any(indicator in text for indicator in indicators)


def extract_followup_weather_city(user_text, history_context):
    if not history_is_waiting_for_weather_city(history_context):
        return None

    text = normalize_city_lookup_text(user_text)
    tokens = text.split()

    if len(tokens) > 4:
        return None

    return extract_city_from_text(user_text)


def is_session_reference_message(user_text):
    text = normalize_basic_text(user_text)

    markers = [
        "az önce",
        "az once",
        "önceki",
        "onceki",
        "son söylediğim",
        "son soyledigim",
        "son yazdığım",
        "son yazdigim",
        "ona göre",
        "ona gore",
        "buna göre",
        "buna gore",
        "devam et",
        "devam",
        "aynı şekilde",
        "ayni sekilde",
        "ne demiştim",
        "ne demistim",
        "hatırlıyor musun",
        "hatirliyor musun",
        "according to that",
        "based on that",
        "continue",
        "what did i say",
        "previous message",
        "last message"
    ]

    return any(marker in text for marker in markers)


def get_source_label(language):
    if language == "en":
        return "Source"

    return "Kaynak"


def get_knowledge_not_found_message(language):
    if language == "en":
        return (
            "I could not find enough information about this topic in my knowledge base. "
            "I can currently answer based on my knowledge base about daily planning, focus, breaks, planner usage, "
            "daily routines, simple meal ideas and weather-based preparation."
        )

    return (
        "Bu konuda bilgi tabanımda yeterli bilgi bulamadım. "
        "Şu an yalnızca günlük planlama, odaklanma, mola yönetimi, plan defteri, günlük rutin, "
        "basit yemek fikirleri ve hava durumuna göre hazırlık konularındaki bilgi tabanıma göre cevap verebilirim."
    )


def get_weather_missing_city_message(language):
    if language == "en":
        return "I can check the weather, but I need the city name first."

    return "Hava durumunu kontrol edebilmem için şehir adını yazar mısın?"


def build_simple_weather_answer(weather_data, language):
    city = weather_data.get("city") or "belirtilen şehir"
    temperature = weather_data.get("temperature_c")
    feels_like = weather_data.get("feels_like_c")
    humidity = weather_data.get("humidity_percent")
    precipitation = weather_data.get("precipitation_mm")
    rain = weather_data.get("rain_mm")
    wind = weather_data.get("wind_speed_kmh")
    description = weather_data.get("weather_description")

    if language == "en":
        parts = [f"Current weather for {city}:"]

        if temperature is not None:
            parts.append(f"- Temperature: {temperature}°C")

        if feels_like is not None:
            parts.append(f"- Feels like: {feels_like}°C")

        if humidity is not None:
            parts.append(f"- Humidity: {humidity}%")

        if wind is not None:
            parts.append(f"- Wind: {wind} km/h")

        if precipitation is not None:
            parts.append(f"- Precipitation: {precipitation} mm")

        if rain is not None:
            parts.append(f"- Rain: {rain} mm")

        if description:
            parts.append(f"- Condition: {description}")

        parts.append("You can use this to decide whether you need an umbrella, jacket or lighter clothing.")

        return "\n".join(parts)

    parts = [f"{city} için güncel hava durumu:"]

    if temperature is not None:
        parts.append(f"- Sıcaklık: {temperature}°C")

    if feels_like is not None:
        parts.append(f"- Hissedilen: {feels_like}°C")

    if humidity is not None:
        parts.append(f"- Nem: %{humidity}")

    if wind is not None:
        parts.append(f"- Rüzgar: {wind} km/sa")

    if precipitation is not None:
        parts.append(f"- Yağış: {precipitation} mm")

    if rain is not None:
        parts.append(f"- Yağmur: {rain} mm")

    if description:
        parts.append(f"- Durum: {description}")

    parts.append("Buna göre şemsiye, mont/ceket veya daha hafif kıyafet ihtiyacını değerlendirebilirsin.")

    return "\n".join(parts)


def get_safe_session(user_id, username):
    try:
        session_id, history_context = get_session_and_history(
            user_id=user_id,
            username=username
        )

        return session_id, history_context

    except Exception as error:
        print(f"Session/history hatası: {error}")

        session_id = f"{user_id}_temporary"
        history_context = ""

        return session_id, history_context


def safe_save_interaction(session_id, user_id, username, first_name, question, answer):
    try:
        save_interaction(
            session_id=session_id,
            user_id=user_id,
            username=username,
            first_name=first_name,
            question=question,
            answer=answer
        )

        print("Interaction DB kaydı yapıldı.")

    except Exception as error:
        print(f"Interaction DB kayıt hatası: {error}")


def safe_save_rag_log(session_id, user_id, username, question, source_files, matched_text):
    try:
        save_rag_log(
            session_id=session_id,
            user_id=user_id,
            username=username,
            question=question,
            source_files=source_files,
            matched_text=matched_text
        )

        print("RAG log kaydedildi.")

    except Exception as error:
        print(f"RAG log kayıt hatası: {error}")


def safe_save_service_log(
    session_id,
    user_id,
    username,
    service_name,
    request_data,
    response_data,
    duration_ms=None
):
    try:
        save_service_log(
            session_id=session_id,
            user_id=user_id,
            username=username,
            service_name=service_name,
            request_data=request_data,
            response_data=response_data,
            duration_ms=duration_ms
        )

        print(f"Service log kaydedildi: {service_name} - {duration_ms} ms")

    except TypeError:
        try:
            save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name=service_name,
                request_data=request_data,
                response_data=response_data
            )

            print(f"Service log kaydedildi: {service_name}")

        except Exception as error:
            print(f"Service log kayıt hatası: {error}")

    except Exception as error:
        print(f"Service log kayıt hatası: {error}")


def generate_general_fallback_answer(user_text, history_context, language):
    ai_answer, ai_error = call_with_timeout(
        func=lambda: ask_openrouter_general_fallback(
            user_text=user_text,
            history_context=history_context
        ),
        timeout_seconds=15
    )

    if ai_error or is_bad_ai_answer(ai_answer):
        return get_knowledge_not_found_message(language)

    return ai_answer

def normalize_location_text(text):
    text = str(text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_location_lookup_text(text):
    text = str(text or "").lower()

    replacements = {
        "ı": "i",
        "i̇": "i",
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


def clean_extracted_city(city):
    city = normalize_location_text(city)

    city = re.sub(
        r"\b(bugün|bugun|yarın|yarin|şu an|su an|şimdi|simdi|today|tomorrow|now)\b",
        "",
        city,
        flags=re.IGNORECASE
    )

    city = re.sub(
        r"\b(hava|weather|durumu|nasıl|nasil|how|is|the|in|for|için|icin|yağmur|yagmur|rain|sıcaklık|sicaklik|temperature)\b",
        "",
        city,
        flags=re.IGNORECASE
    )

    city = re.sub(r"\s+", " ", city).strip()
    city = city.strip(" '’`.,?!")

    if not city:
        return None

    if len(city) < 2:
        return None

    # "yaşadığım yer", "bulunduğum yer" gibi ifadeler şehir değildir.
    invalid_location_phrases = [
        "yaşadığım yerde",
        "yasadigim yerde",
        "yaşadığım yer",
        "yasadigim yer",
        "bulunduğum yerde",
        "bulundugum yerde",
        "bulunduğum yer",
        "bulundugum yer",
        "burada",
        "burası",
        "burasi",
        "konumum"
    ]

    normalized_city = normalize_location_lookup_text(city)

    if any(phrase in normalized_city for phrase in invalid_location_phrases):
        return None

    return city


def extract_city_from_weather_sentence(user_text):
    """
    Şehir listesi kullanmadan cümleden lokasyon çıkarmaya çalışır.
    Örnekler:
    - Paris'te hava nasıl? -> Paris
    - Londra’da hava nasıl? -> Londra
    - weather in New York -> New York
    """

    text = normalize_location_text(user_text)

    patterns = [
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)['’]?(?:da|de|ta|te)\s+(?:hava|yağmur|yagmur|sıcaklık|sicaklik|weather|rain|temperature)",
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)\s+(?:hava durumu|weather)",
        r"(?:weather in|weather for|how is the weather in)\s+(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+)",
        r"(?P<city>[A-Za-zÇĞİÖŞÜçğıöşü\s]+?)\s+(?:için|icin|for)\s+(?:hava|weather|sıcaklık|sicaklik|temperature)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            city = clean_extracted_city(match.group("city"))

            if city:
                return city

    return None


# Eski kodda generate_ai_answer içinde extract_city_from_text çağrısı kalmış olabilir.
# Bu fonksiyonu güvenli hale getiriyoruz.
def extract_city_from_text(user_text):
    return extract_city_from_weather_sentence(user_text)


def history_is_waiting_for_weather_city(history_context):
    text = normalize_location_lookup_text(history_context)

    indicators = [
        "sehir adini yazar misin",
        "hava durumunu kontrol edebilmem icin sehir",
        "i need the city name",
        "need the city name",
        "city name first"
    ]

    return any(indicator in text for indicator in indicators)


def extract_followup_weather_city(user_text, history_context):
    """
    Kullanıcı önce 'hava nasıl?' dedi, bot şehir sordu.
    Sonra kullanıcı sadece 'Paris', 'New York', 'İstanbul' yazarsa bunu şehir kabul eder.
    """

    if not history_is_waiting_for_weather_city(history_context):
        return None

    text = normalize_location_text(user_text)
    words = text.split()

    if 1 <= len(words) <= 5:
        return clean_extracted_city(text)

    return None


def extract_weather_city_with_ai(user_text, history_context):
    """
    Asıl akıllı yol:
    AI/tool router şehir parametresini çıkarmaya çalışır.
    """

    tool_call = detect_tool_call(
        user_text=user_text,
        history_context=history_context
    )

    if tool_call.get("tool") == "weather":
        city = tool_call.get("city")

        if city:
            return clean_extracted_city(city), tool_call

    return None, tool_call


def generate_ai_answer(user_text, session_id, history_context, user_id, username):
    print("generate_ai_answer çalıştı.")
    print(f"Gelen mesaj: {user_text}")

    language = detect_user_language(user_text)

    if is_meaningless_input(user_text):
        return {
            "answer": get_meaningless_input_answer(language),
            "source_type": "invalid_or_meaningless_input",
            "sources": []
        }

    basic_answer = get_basic_conversation_answer(
        user_text=user_text,
        language=language
    )

    if basic_answer:
        return {
            "answer": basic_answer,
            "source_type": "basic_conversation",
            "sources": []
        }

    city_from_text = extract_city_from_weather_sentence(user_text)

    followup_city = extract_followup_weather_city(
    user_text=user_text,
    history_context=history_context
    )

    is_weather_request = looks_like_weather_question(user_text) or followup_city is not None

    if is_weather_request:
        print("Hava durumu sorusu algılandı.")

        city = city_from_text or followup_city
        tool_call = {}

        if not city:
            print("Şehir regex ile yakalanamadı. AI tool router çalışacak.")
            ai_city, tool_call = extract_weather_city_with_ai(
                user_text=user_text,
                history_context=history_context
            )
            if ai_city:
                city = ai_city
            if city:
                print(f"Şehir yakalandı: {city}")


            safe_save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name="deterministic_weather_city_detection",
                request_data=to_json_text({
                    "user_text": user_text,
                    "history_context_used": bool(history_context)
                }),
                response_data=to_json_text({
                    "city": city
                }),
                duration_ms=0
            )

        else:
            print("Şehir direkt yakalanamadı. Tool calling çalışacak.")

            tool_start = now_ms()

            tool_call = detect_tool_call(user_text, history_context)

            tool_duration = elapsed_ms(tool_start)

            safe_save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name="tool_decision_weather",
                request_data=to_json_text({
                    "user_text": user_text,
                    "history_context_used": bool(history_context)
                }),
                response_data=to_json_text(tool_call),
                duration_ms=tool_duration
            )

            print("Tool call sonucu:")
            print(tool_call)

            if tool_call.get("tool") == "weather":
                city = tool_call.get("city")

        if not city:
            return {
                "answer": get_weather_missing_city_message(language),
                "source_type": "weather_missing_city",
                "sources": []
            }

        weather_result = get_weather_by_city(city)

        for log in weather_result.get("logs", []):
            request_log = log.get("request")
            response_log = log.get("response")
            duration_ms = log.get("duration_ms")

            provider = "weather_api"

            if isinstance(request_log, dict):
                provider = request_log.get("provider", provider)

            safe_save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name=provider,
                request_data=to_json_text(request_log),
                response_data=to_json_text(response_log),
                duration_ms=duration_ms
            )

        if weather_result["success"]:
            weather_context = to_json_text(weather_result["data"])

            llm_start = now_ms()

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

            llm_duration = elapsed_ms(llm_start)

            if is_bad_ai_answer(ai_answer):
                ai_answer = build_simple_weather_answer(
                    weather_data=weather_result["data"],
                    language=language
                )

            safe_save_service_log(
                session_id=session_id,
                user_id=user_id,
                username=username,
                service_name="weather_answer_llm",
                request_data=to_json_text({
                    "user_text": user_text,
                    "weather_context": weather_result["data"]
                }),
                response_data=ai_answer,
                duration_ms=llm_duration
            )

            return {
                "answer": ai_answer,
                "source_type": "weather_tool",
                "sources": [weather_result["data"]]
            }

        return {
            "answer": weather_result["message"],
            "source_type": "weather_error",
            "sources": []
        }

    if is_session_reference_message(user_text):
        print("Session referanslı mesaj algılandı.")

        if history_context and history_context.strip():
            ai_answer = generate_general_fallback_answer(
                user_text=user_text,
                history_context=history_context,
                language=language
            )

            return {
                "answer": ai_answer,
                "source_type": "session_context_answer",
                "sources": []
            }

        if language == "en":
            answer = "I could not find enough previous conversation context for this session."
        else:
            answer = "Bu session içinde yeterli önceki konuşma bulamadım."

        return {
            "answer": answer,
            "source_type": "session_context_missing",
            "sources": []
        }

    print("Hava durumu sorusu değil. Önce RAG çalışacak.")

    rag_search = search_knowledge_base(user_text)

    print("RAG araması yapıldı.")
    print(rag_search)

    if rag_search["found"]:
        rag_context = format_rag_context(rag_search["results"])
        rag_sources = get_rag_sources_json(rag_search["results"])

        print("RAG sonucu bulundu.")
        print(rag_sources)

        safe_save_rag_log(
            session_id=session_id,
            user_id=user_id,
            username=username,
            question=user_text,
            source_files=rag_sources,
            matched_text=rag_context
        )

        ai_answer, rag_ai_error = call_with_timeout(
            func=lambda: ask_openrouter_with_rag(
                user_text=user_text,
                rag_context=rag_context,
                history_context=history_context
            ),
            timeout_seconds=45
        )

        if rag_ai_error or is_bad_ai_answer(ai_answer):
            print(f"RAG AI cevabı üretilemedi, direkt bilgi tabanı cevabına düşüldü: {rag_ai_error}")

            ai_answer = build_direct_rag_answer(
                question=user_text,
                rag_results=rag_search["results"],
                language=language,
                history_context=history_context
            )
            if is_bad_ai_answer(ai_answer):
                 if language == "en":
                    ai_answer = (
                "I found relevant information in the knowledge base, but I could not generate a proper answer right now. "
                "Please try again in a few seconds."
            )
            else:
                ai_answer = (
                "Bilgi tabanında ilgili bilgi buldum fakat şu anda düzgün bir cevap üretemedim. "
                "Birkaç saniye sonra tekrar dener misin?"
            )
            return {
                "answer": ai_answer,
                "source_type": "rag_direct_fallback",
                "sources": rag_search["results"]
            }

        source_label = get_source_label(language)
        first_source = rag_search["results"][0]["source_file"]

        if f"{source_label}:" not in ai_answer and "Kaynak:" not in ai_answer and "Source:" not in ai_answer:
            ai_answer = f"{ai_answer}\n\n{source_label}: {first_source}"

        print("RAG cevabı AI ile üretildi.")

        return {
            "answer": ai_answer,
            "source_type": "rag_ai",
            "sources": rag_search["results"]
        }

    print("RAG sonucu bulunamadı. Güvenli genel fallback çalışacak.")

    ai_answer = generate_general_fallback_answer(
        user_text=user_text,
        history_context=history_context,
        language=language
    )

    return {
        "answer": ai_answer,
        "source_type": "general_ai_fallback",
        "sources": []
    }


def process_text_message(user_id, username, first_name, user_text):
    session_id, history_context = get_safe_session(
        user_id=user_id,
        username=username
    )

    result = generate_ai_answer(
        user_text=user_text,
        session_id=session_id,
        history_context=history_context,
        user_id=user_id,
        username=username
    )

    if result is None:
        result = {
            "answer": "Cevap üretirken bir sorun oluştu. Lütfen tekrar dener misin?",
            "source_type": "internal_none_result",
            "sources": []
        }

    answer = clean_final_answer(result.get("answer", "Cevap alınamadı."))

    safe_save_interaction(
        session_id=session_id,
        user_id=user_id,
        username=username,
        first_name=first_name,
        question=user_text,
        answer=answer
    )

    return {
        "success": True,
        "session_id": session_id,
        "input_text": user_text,
        "answer": answer,
        "source_type": result.get("source_type"),
        "sources": result.get("sources", [])
    }


def process_voice_message(user_id, username, first_name, audio_path, language="tr-TR"):
    session_id, history_context = get_safe_session(
        user_id=user_id,
        username=username
    )

    stt_result = speech_to_text(
        audio_path=audio_path,
        language=language
    )

    if not stt_result["success"]:
        answer = stt_result["message"]

        safe_save_interaction(
            session_id=session_id,
            user_id=user_id,
            username=username,
            first_name=first_name,
            question="[VOICE_STT_FAILED]",
            answer=answer
        )

        return {
            "success": False,
            "session_id": session_id,
            "transcript": "",
            "answer": answer,
            "voice_path": None,
            "source_type": "stt_failed",
            "sources": []
        }

    transcript = stt_result["text"]
    detected_language = detect_user_language(transcript)

    result = generate_ai_answer(
        user_text=transcript,
        session_id=session_id,
        history_context=history_context,
        user_id=user_id,
        username=username
    )

    if result is None:
        result = {
            "answer": "Cevap üretirken bir sorun oluştu. Lütfen tekrar dener misin?",
            "source_type": "internal_none_result",
            "sources": []
        }

    answer = result.get("answer", "Cevap alınamadı.")

    safe_save_interaction(
        session_id=session_id,
        user_id=user_id,
        username=username,
        first_name=first_name,
        question=f"[VOICE] {transcript}",
        answer=answer
    )

    tts_language = "en" if detected_language == "en" else "tr"

    tts_result = text_to_speech(
        text=answer,
        language=tts_language
    )

    voice_path = None

    if tts_result["success"]:
        voice_path = tts_result["voice_path"]

    return {
        "success": True,
        "session_id": session_id,
        "transcript": transcript,
        "answer": answer,
        "voice_path": voice_path,
        "source_type": result.get("source_type"),
        "sources": result.get("sources", [])
    }


def process_image_message(user_id, username, first_name, image_path, caption=""):
    session_id, history_context = get_safe_session(
        user_id=user_id,
        username=username
    )

    try:
        answer = analyze_image_with_openrouter(
            image_path=image_path,
            caption=caption,
            history_context=history_context
        )

    except Exception as error:
        print(f"Image analyze error: {error}")
        answer = "Fotoğrafı analiz ederken bir hata oluştu. Lütfen tekrar dener misin?"

    safe_save_interaction(
        session_id=session_id,
        user_id=user_id,
        username=username,
        first_name=first_name,
        question=f"[PHOTO] {caption}",
        answer=answer
    )

    return {
        "success": True,
        "session_id": session_id,
        "caption": caption,
        "answer": answer,
        "source_type": "vision",
        "sources": []
    }