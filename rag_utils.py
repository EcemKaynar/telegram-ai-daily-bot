import json
import os
import re


KNOWLEDGE_BASE_DIR = "knowledge_base"


STOPWORDS = {
    # Turkish
    "ve", "veya", "ile", "için", "bu", "şu", "o", "bir", "de", "da",
    "mi", "mı", "mu", "mü", "ne", "nasıl", "neden", "ben", "sen",
    "bana", "sana", "çok", "az", "gibi", "ama", "fakat", "ise",
    "şey", "şeyi", "şunu", "bunu",

    # English
    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on",
    "at", "is", "are", "am", "be", "being", "been", "i", "you",
    "me", "my", "your", "we", "they", "it", "this", "that",
    "these", "those", "can", "could", "would", "should", "do",
    "does", "did", "with", "about", "hi", "hello", "please", "tell"
}


ENGLISH_EXPANSIONS = {
    "plan": ["planlama", "günlük", "görev", "ajanda", "defter", "rutin"],
    "planning": ["planlama", "günlük", "görev", "ajanda", "defter", "rutin"],
    "day": ["günlük", "gün", "rutin", "planlama"],
    "daily": ["günlük", "rutin", "planlama"],

    "routine": ["rutin", "günlük", "sabah", "alışkanlık"],
    "morning": ["sabah", "rutin", "günlük"],
    "habit": ["alışkanlık", "rutin", "günlük"],

    "focus": ["odak", "çalışma", "mola", "dikkat"],
    "productive": ["verimli", "odak", "çalışma", "görev"],
    "productivity": ["verimli", "odak", "çalışma", "görev"],
    "unproductive": ["verimsiz", "odak", "başlamak", "görev"],
    "motivation": ["motivasyon", "isteksiz", "başlamak", "görev"],

    "break": ["mola", "kahve", "çay", "dinlenme"],
    "coffee": ["kahve", "mola", "çalışma"],
    "tea": ["çay", "mola", "çalışma"],
    "rest": ["mola", "dinlenme"],

    "work": ["çalışma", "görev", "odak"],
    "study": ["çalışma", "odak", "görev"],
    "task": ["görev", "planlama", "başlamak"],
    "tasks": ["görev", "planlama", "başlamak"],

    "meal": ["yemek", "basit", "rutin"],
    "meals": ["yemek", "basit", "rutin"],
    "food": ["yemek", "basit", "rutin"],
    "eat": ["yemek", "basit", "rutin"],
    "healthy": ["yemek", "basit", "rutin"],
    "diet": ["yemek", "basit", "rutin"],
    "nutrition": ["yemek", "basit", "rutin"],

    "weather": ["hava", "şemsiye", "yağmur", "hazırlık"],
    "rain": ["yağmur", "şemsiye", "hava"],
    "umbrella": ["şemsiye", "yağmur", "hava"],
    "cold": ["soğuk", "hava", "ceket"],
    "hot": ["sıcak", "hava"]
}


TURKISH_EXPANSIONS = {
    "verimsiz": ["odak", "başlamak", "görev", "motivasyon", "planlama"],
    "isteksiz": ["motivasyon", "başlamak", "görev", "odak"],
    "motivasyon": ["isteksiz", "başlamak", "görev", "odak"],
    "odak": ["çalışma", "mola", "dikkat", "verimli"],
    "plan": ["planlama", "günlük", "görev", "ajanda", "defter"],
    "planlama": ["günlük", "görev", "ajanda", "defter"],
    "defter": ["plan", "planlama", "ajanda", "görev"],
    "ajanda": ["plan", "planlama", "defter", "görev"],
    "mola": ["kahve", "çay", "dinlenme", "odak"],
    "kahve": ["mola", "çalışma", "odak"],
    "çay": ["mola", "çalışma", "odak"],
    "rutin": ["günlük", "sabah", "alışkanlık"],
    "sabah": ["rutin", "günlük"],
    "yemek": ["basit", "rutin", "günlük"],
    "hava": ["şemsiye", "yağmur", "hazırlık"],
    "şemsiye": ["hava", "yağmur"]
}


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ı", "i")

    return text


def tokenize(text):
    normalized = normalize_text(text)

    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", normalized)

    clean_tokens = []

    for token in tokens:
        token = token.strip().lower()

        if not token:
            continue

        if token in STOPWORDS:
            continue

        if len(token) < 2:
            continue

        clean_tokens.append(token)

    return clean_tokens


def detect_user_language(text):
    text_lower = str(text or "").lower()
    tokens = tokenize(text_lower)

    turkish_chars = ["ç", "ğ", "ı", "ö", "ş", "ü"]

    if any(char in text_lower for char in turkish_chars):
        return "tr"

    english_markers = {
        "what", "should", "plan", "day", "daily", "routine", "healthy",
        "diet", "how", "focus", "productive", "weather", "coffee",
        "break", "meal", "food", "work", "study"
    }

    turkish_markers = {
        "bugün", "gun", "gün", "nasıl", "yapmalıyım", "verimli",
        "mola", "kahve", "planlama", "hava", "rutin", "yemek"
    }

    if any(token in english_markers for token in tokens):
        return "en"

    if any(token in turkish_markers for token in tokens):
        return "tr"

    return "tr"


def expand_query_tokens(tokens):
    expanded = set(tokens)

    for token in tokens:
        if token in ENGLISH_EXPANSIONS:
            expanded.update(ENGLISH_EXPANSIONS[token])

        if token in TURKISH_EXPANSIONS:
            expanded.update(TURKISH_EXPANSIONS[token])

    return list(expanded)


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_document_into_chunks(text):
    parts = re.split(r"\n\s*\n", text)

    chunks = []

    for part in parts:
        clean_part = part.strip()

        if clean_part:
            chunks.append(clean_part)

    if not chunks and text.strip():
        chunks.append(text.strip())

    return chunks


def load_knowledge_base():
    documents = []

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return documents

    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)

        try:
            content = read_text_file(file_path)
            chunks = split_document_into_chunks(content)

            for index, chunk in enumerate(chunks):
                documents.append({
                    "source_file": file_name,
                    "chunk_id": f"{file_name}-{index + 1}",
                    "text": chunk
                })

        except Exception as error:
            print(f"Bilgi tabanı dosyası okunamadı: {file_name} - {error}")

    return documents


def calculate_score(question, chunk_text):
    question_tokens = tokenize(question)
    expanded_tokens = expand_query_tokens(question_tokens)
    chunk_tokens = tokenize(chunk_text)

    question_token_set = set(question_tokens)
    expanded_token_set = set(expanded_tokens)
    chunk_token_set = set(chunk_tokens)

    exact_matches = question_token_set.intersection(chunk_token_set)
    expanded_matches = expanded_token_set.intersection(chunk_token_set)

    score = 0

    score += len(exact_matches) * 3
    score += len(expanded_matches)

    question_lower = normalize_text(question)
    chunk_lower = normalize_text(chunk_text)

    phrase_bonus_map = {
        "healthy diet": ["yemek", "basit", "rutin"],
        "daily plan": ["günlük", "planlama", "görev"],
        "plan for a day": ["günlük", "planlama", "görev"],
        "coffee break": ["kahve", "mola"],
        "focus": ["odak", "çalışma"],
        "morning routine": ["sabah", "rutin"]
    }

    for phrase, related_words in phrase_bonus_map.items():
        if phrase in question_lower:
            for word in related_words:
                if word in chunk_lower:
                    score += 3

    return score


def search_knowledge_base(question, max_results=3, min_score=2):
    documents = load_knowledge_base()

    scored_results = []

    for document in documents:
        score = calculate_score(question, document["text"])

        if score >= min_score:
            scored_results.append({
                "source_file": document["source_file"],
                "chunk_id": document["chunk_id"],
                "score": score,
                "text": document["text"]
            })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    results = scored_results[:max_results]

    return {
        "found": len(results) > 0,
        "results": results,
        "query_tokens": tokenize(question),
        "expanded_tokens": expand_query_tokens(tokenize(question))
    }


def format_rag_context(rag_results):
    context_parts = []

    for result in rag_results:
        context_parts.append(
            f"Source file: {result['source_file']}\n"
            f"Chunk ID: {result['chunk_id']}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_rag_sources_json(rag_results):
    sources = []

    for result in rag_results:
        sources.append({
            "source_file": result["source_file"],
            "chunk_id": result["chunk_id"],
            "score": result["score"]
        })

    return json.dumps(sources, ensure_ascii=False)


def build_english_direct_answer(question, best_result):
    source_file = best_result.get("source_file", "")
    question_lower = normalize_text(question)

    if "planlama" in source_file:
        return (
            "According to my knowledge base, daily planning should not be about filling the whole day completely. "
            "It should make the day easier to manage. A good starting point is to choose a small task first, "
            "then define the 3 most important tasks of the day. Using a planner or agenda can also help you see "
            "your tasks more clearly.\n\n"
            f"Source: {source_file}"
        )

    if "mola" in source_file or "odak" in source_file:
        return (
            "According to my knowledge base, taking a break does not mean being unproductive. "
            "A short coffee or tea break can help you return to work with better focus. "
            "After the break, start with one small and clear task instead of trying to do everything at once.\n\n"
            f"Source: {source_file}"
        )

    if "gunluk" in source_file or "rutin" in source_file:
        if "diet" in question_lower or "healthy" in question_lower or "nutrition" in question_lower:
            return (
                "My knowledge base does not include a detailed healthy diet program. "
                "It only contains general daily routine and simple meal idea guidance. "
                "Based on that, I can suggest keeping meals simple and manageable instead of creating a complicated plan. "
                "For a personal medical or nutrition diet, it would be better to consult a professional.\n\n"
                f"Source: {source_file}"
            )

        return (
            "According to my knowledge base, a simple daily routine can make the day easier to manage. "
            "A basic morning routine, simple meal ideas and small daily habits can help you feel more organized.\n\n"
            f"Source: {source_file}"
        )

    return (
        "According to my knowledge base:\n"
        f"{best_result.get('text', '')}\n\n"
        f"Source: {source_file}"
    )


def build_turkish_direct_answer(question, best_result):
    source_file = best_result.get("source_file", "")
    text = best_result.get("text", "")

    return (
        "Bilgi tabanıma göre:\n"
        f"{text}\n\n"
        f"Kaynak: {source_file}"
    )


def build_direct_rag_answer(question, rag_results, language=None):
    detected_language = language or detect_user_language(question)

    if not rag_results:
        if detected_language == "en":
            return (
                "I could not find enough information about this topic in my knowledge base. "
                "I can currently answer based on my knowledge base about daily planning, focus, breaks, simple routines, "
                "simple meal ideas and weather-based preparation."
            )

        return (
            "Bu konuda bilgi tabanımda yeterli bilgi bulamadım. "
            "Şu an yalnızca günlük planlama, odaklanma, mola yönetimi, plan defteri, günlük rutin, "
            "basit yemek fikirleri ve hava durumuna göre hazırlık konularındaki bilgi tabanıma göre cevap verebilirim."
        )

    best_result = rag_results[0]

    if detected_language == "en":
        return build_english_direct_answer(question, best_result)

    return build_turkish_direct_answer(question, best_result)