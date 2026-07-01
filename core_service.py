import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from openrouter_client import (
    ask_openrouter_with_context,
    ask_openrouter_with_tool_result,
    analyze_image_with_openrouter,
    ask_openrouter_with_rag
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
    ]

    return any(phrase in cleaned_lower for phrase in bad_phrases)


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

    # Sadece kısa selamlaşmaları greeting sayıyoruz.
    # Böylece "hi can you tell me a healthy diet" gibi cümleleri yanlışlıkla selam sanmaz.
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
        "wind",
        "cold",
        "hot"
    ]

    return any(word in text for word in weather_words)


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


def generate_ai_answer(user_text, session_id, history_context, user_id, username):
    print("generate_ai_answer çalıştı.")
    print(f"Gelen mesaj: {user_text}")

    language = detect_user_language(user_text)

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
                timeout_seconds=15
            )

            if rag_ai_error or is_bad_ai_answer(ai_answer):
                print(f"RAG AI cevabı üretilemedi, direkt bilgi tabanı cevabına düşüldü: {rag_ai_error}")

                ai_answer = build_direct_rag_answer(
                    question=user_text,
                    rag_results=rag_search["results"],
                    language=language
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

        print("RAG sonucu bulunamadı.")

        return {
            "answer": get_knowledge_not_found_message(language),
            "source_type": "knowledge_not_found",
            "sources": []
        }

    print("Hava durumu sorusu algılandı. Tool calling çalışacak.")

    tool_call = detect_tool_call(user_text, history_context)

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
            try:
                save_service_log(
                    session_id=session_id,
                    user_id=user_id,
                    username=username,
                    service_name="open_meteo_weather",
                    request_data=to_json_text(log.get("request")),
                    response_data=to_json_text(log.get("response"))
                )
            except Exception as error:
                print(f"Service log kayıt hatası: {error}")

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

    return {
        "answer": get_weather_missing_city_message(language),
        "source_type": "weather_missing_city",
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

    answer = result["answer"]

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

    answer = result["answer"]

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