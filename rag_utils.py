import json
import os
import re


KNOWLEDGE_BASE_DIR = "knowledge_base"
RAG_INDEX_FILE = os.getenv("RAG_INDEX_FILE", "knowledge_index.json")
DEFAULT_RAG_THRESHOLD = int(os.getenv("RAG_THRESHOLD", "8"))
DEFAULT_RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))


STOPWORDS = {
    "ve", "veya", "ile", "için", "bu", "şu", "o", "bir", "de", "da",
    "mi", "mı", "mu", "mü", "ne", "nasıl", "neden", "ben", "sen",
    "bana", "sana", "çok", "az", "gibi", "ama", "fakat", "ise",
    "şey", "şeyi", "şunu", "bunu", "şeklinde", "olarak",

    "a", "an", "the", "and", "or", "for", "to", "of", "in", "on",
    "at", "is", "are", "am", "be", "being", "been", "i", "you",
    "me", "my", "your", "we", "they", "it", "this", "that",
    "these", "those", "can", "could", "would", "should", "do",
    "does", "did", "with", "about", "hi", "hello", "please", "tell"
}


ENGLISH_EXPANSIONS = {
    "plan": ["planlama", "günlük", "görev", "ajanda", "defter", "rutin", "öncelik"],
    "planning": ["planlama", "günlük", "görev", "ajanda", "defter", "rutin", "öncelik"],
    "day": ["günlük", "gün", "rutin", "planlama"],
    "daily": ["günlük", "rutin", "planlama"],
    "planner": ["ajanda", "plan", "planlama", "defter"],
    "agenda": ["ajanda", "plan", "planlama", "defter"],

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

    "meal": ["yemek", "basit", "rutin"],
    "meals": ["yemek", "basit", "rutin"],
    "food": ["yemek", "basit", "rutin"],
    "healthy": ["yemek", "basit", "rutin"],
    "diet": ["yemek", "basit", "rutin"],

    "weather": ["hava", "şemsiye", "yağmur", "hazırlık"],
    "rain": ["yağmur", "şemsiye", "hava"],
    "umbrella": ["şemsiye", "yağmur", "hava"],
}


TURKISH_EXPANSIONS = {
    "verimsiz": ["odak", "başlamak", "görev", "motivasyon", "planlama"],
    "isteksiz": ["motivasyon", "başlamak", "görev", "odak"],
    "motivasyon": ["isteksiz", "başlamak", "görev", "odak"],
    "odak": ["çalışma", "mola", "dikkat", "verimli"],
    "odaklanma": ["çalışma", "mola", "dikkat", "verimli"],
    "plan": ["planlama", "günlük", "görev", "ajanda", "defter", "öncelik"],
    "planlama": ["günlük", "görev", "ajanda", "defter", "öncelik"],
    "defter": ["plan", "planlama", "ajanda", "görev"],
    "ajanda": ["plan", "planlama", "defter", "görev"],
    "görev": ["planlama", "öncelik", "günlük"],
    "mola": ["kahve", "çay", "dinlenme", "odak"],
    "kahve": ["mola", "çalışma", "odak"],
    "çay": ["mola", "çalışma", "odak"],
    "rutin": ["günlük", "sabah", "alışkanlık"],
    "sabah": ["rutin", "günlük"],
    "yemek": ["basit", "rutin", "günlük"],
    "hava": ["şemsiye", "yağmur", "hazırlık"],
    "şemsiye": ["hava", "yağmur"],
}


TOPIC_QUERY_KEYWORDS = {
    "daily_planning": [
        "günlük plan", "gunluk plan", "planlama", "plan yapmak", "günümü planla",
        "günümü nasıl planlamalıyım", "ajanda", "plan defteri", "öncelik",
        "görev", "yapılacaklar", "daily plan", "planning", "planner", "agenda"
    ],
    "focus_productivity": [
        "odak", "odaklanamıyorum", "verimli", "verimsiz", "motivasyon",
        "dikkatim dağılıyor", "çalışmaya başlayamıyorum", "focus",
        "productive", "productivity", "unproductive", "motivation"
    ],
    "break_management": [
        "mola", "kahve molası", "çay molası", "dinlenme", "ara vermek",
        "break", "coffee break", "tea break", "rest"
    ],
    "daily_routine": [
        "rutin", "sabah rutini", "akşam rutini", "alışkanlık", "güne başlamak",
        "routine", "morning routine", "habit"
    ],
    "simple_meals": [
        "yemek", "basit yemek", "ne yiyebilirim", "pratik öğün", "sağlıklı yemek",
        "meal", "food", "simple meal", "what can i eat"
    ],
    "weather_preparation": [
        "hava", "yağmur", "şemsiye", "mont", "ceket", "sıcaklık",
        "weather", "rain", "umbrella", "jacket"
    ],
    "safe_fallback": [
        "bitcoin", "coin", "borsa", "yatırım", "hukuk", "avukat", "doktor",
        "hastalık", "investment", "medical", "legal"
    ],
    "help": [
        "ne yapabiliyorsun", "neler yapabilirsin", "yardım", "help",
        "what can you do", "how can you help"
    ]
}


BAD_USER_VISIBLE_MARKERS = [
    "sayfa / page",
    "rag knowledge document",
    "topic:",
    "keywords:",
    "konu / topic",
    "anahtar kelimeler",
    "generated for telegram",
    "threshold",
    "top-k",
    "top 3 chunk",
    "chunk",
    "jwt",
    "middleware",
    "api key"
]


HELP_SECTION_MARKERS = [
    "ne yapabiliyorsun",
    "neler yapabilirsin",
    "asistan kendini",
    "bot kendini",
    "yardım mesajı",
    "kullanıcı ne yapabiliyorsun",
    "what can you do",
    "assistant introduces itself"
]


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ı", "i")
    return text


def clean_user_visible_text(text):
    text = str(text or "")

    text = re.sub(r"Sayfa\s*/\s*Page\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"RAG knowledge document\s*-\s*generated.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"topic\s*:\s*[a-zA-Z0-9_\-]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"keywords\s*:\s*[^.\n]+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Konu\s*/\s*Topic.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Anahtar Kelimeler\s*/\s*Keywords.*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


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
        "break", "meal", "food", "work", "study", "hello", "hi"
    }

    turkish_markers = {
        "bugün", "gun", "gün", "nasıl", "yapmalıyım", "verimli",
        "mola", "kahve", "planlama", "hava", "rutin", "yemek", "merhaba"
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


def infer_query_topic(question):
    question_lower = normalize_text(question)

    topic_scores = {}

    for topic, keywords in TOPIC_QUERY_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if normalize_text(keyword) in question_lower:
                score += 3

        topic_scores[topic] = score

    best_topic = max(topic_scores, key=topic_scores.get)

    if topic_scores[best_topic] == 0:
        return "general"

    return best_topic


def has_bad_user_visible_marker(text):
    lowered = normalize_text(text)

    return any(marker in lowered for marker in BAD_USER_VISIBLE_MARKERS)


def is_help_section(text):
    lowered = normalize_text(text)

    return any(marker in lowered for marker in HELP_SECTION_MARKERS)


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


def load_txt_knowledge_base():
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
                    "topic": "legacy_txt",
                    "keywords": [],
                    "text": chunk
                })

        except Exception as error:
            print(f"Bilgi tabanı dosyası okunamadı: {file_name} - {error}")

    return documents


def load_index_knowledge_base():
    if not os.path.exists(RAG_INDEX_FILE):
        return []

    try:
        with open(RAG_INDEX_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data.get("chunks", [])

    except Exception as error:
        print(f"Knowledge index okunamadı: {error}")
        return []


def load_knowledge_base():
    indexed_chunks = load_index_knowledge_base()

    if indexed_chunks:
        return indexed_chunks

    return load_txt_knowledge_base()


def calculate_score(question, chunk):
    chunk_text = chunk.get("text", "")
    chunk_topic = chunk.get("topic", "")
    chunk_keywords = chunk.get("keywords", [])

    query_topic = infer_query_topic(question)

    question_tokens = tokenize(question)
    expanded_tokens = expand_query_tokens(question_tokens)
    chunk_tokens = tokenize(chunk_text)

    question_token_set = set(question_tokens)
    expanded_token_set = set(expanded_tokens)
    chunk_token_set = set(chunk_tokens)
    keyword_set = set([normalize_text(keyword) for keyword in chunk_keywords])

    exact_matches = question_token_set.intersection(chunk_token_set)
    expanded_matches = expanded_token_set.intersection(chunk_token_set)
    keyword_matches = expanded_token_set.intersection(keyword_set)

    score = 0

    score += len(exact_matches) * 3
    score += len(expanded_matches) * 1
    score += len(keyword_matches) * 2

    if query_topic != "general":
        if chunk_topic == query_topic:
            score += 20
        elif chunk_topic not in ["general", "legacy_txt", ""]:
            score -= 10

    question_lower = normalize_text(question)
    chunk_lower = normalize_text(chunk_text)

    phrase_bonus_map = {
        "günlük planlama": ["günlük", "planlama", "görev", "öncelik"],
        "günlük plan": ["günlük", "planlama", "görev", "öncelik"],
        "planlama nasıl": ["günlük", "planlama", "görev", "öncelik"],
        "daily plan": ["daily", "planning", "task"],
        "daily planning": ["daily", "planning", "task"],
        "plan for a day": ["daily", "planning", "task"],
        "kahve molası": ["kahve", "mola"],
        "coffee break": ["coffee", "break"],
        "sabah rutini": ["sabah", "rutin"],
        "morning routine": ["morning", "routine"],
        "odaklanamıyorum": ["odak", "dikkat", "çalışma"],
        "focus": ["focus", "work", "attention"],
    }

    for phrase, related_words in phrase_bonus_map.items():
        if phrase in question_lower:
            for word in related_words:
                if normalize_text(word) in chunk_lower or normalize_text(word) in normalize_text(chunk_topic):
                    score += 4

    if query_topic != "help" and is_help_section(chunk_text):
        score -= 25

    if query_topic not in ["safe_fallback", "general"] and has_bad_user_visible_marker(chunk_text):
        score -= 15

    return score


def search_knowledge_base(question, max_results=None, min_score=None):
    if max_results is None:
        max_results = DEFAULT_RAG_TOP_K

    if min_score is None:
        min_score = DEFAULT_RAG_THRESHOLD

    documents = load_knowledge_base()
    query_topic = infer_query_topic(question)

    scored_results = []

    for document in documents:
        score = calculate_score(question, document)

        if score >= min_score:
            scored_results.append({
                "source_file": document.get("source_file", ""),
                "chunk_id": document.get("chunk_id", ""),
                "topic": document.get("topic", ""),
                "keywords": document.get("keywords", []),
                "score": score,
                "text": document.get("text", "")
            })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    results = scored_results[:max_results]

    return {
        "found": len(results) > 0,
        "results": results,
        "query_topic": query_topic,
        "query_tokens": tokenize(question),
        "expanded_tokens": expand_query_tokens(tokenize(question)),
        "threshold": min_score,
        "top_k": max_results
    }


def format_rag_context(rag_results):
    context_parts = []

    for result in rag_results:
        clean_text = clean_user_visible_text(result["text"])

        context_parts.append(
            f"Source file: {result['source_file']}\n"
            f"Topic: {result.get('topic', '')}\n"
            f"Content:\n{clean_text}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_rag_sources_json(rag_results):
    sources = []

    for result in rag_results:
        sources.append({
            "source_file": result["source_file"],
            "chunk_id": result["chunk_id"],
            "topic": result.get("topic", ""),
            "score": result["score"]
        })

    return json.dumps(sources, ensure_ascii=False)


def split_sentences(text):
    text = clean_user_visible_text(text)

    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", text)

    sentences = []

    for sentence in raw_sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) < 40:
            continue

        if has_bad_user_visible_marker(sentence):
            continue

        if is_help_section(sentence):
            continue

        sentences.append(sentence)

    return sentences


def extract_relevant_sentences(question, rag_results, max_sentences=5):
    question_tokens = set(expand_query_tokens(tokenize(question)))
    query_topic = infer_query_topic(question)

    scored_sentences = []

    for result in rag_results:
        result_topic = result.get("topic", "")
        sentences = split_sentences(result.get("text", ""))

        for sentence in sentences:
            sentence_tokens = set(tokenize(sentence))
            score = len(question_tokens.intersection(sentence_tokens))

            if result_topic == query_topic:
                score += 5

            if score > 0:
                scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected = []

    for score, sentence in scored_sentences:
        if sentence not in selected:
            selected.append(sentence)

        if len(selected) >= max_sentences:
            break

    return selected


def build_topic_template_answer(question, source_file, language):
    topic = infer_query_topic(question)

    if language == "en":
        templates = {
            "daily_planning": (
                "Daily planning should make the day manageable rather than completely full. "
                "A practical plan can start with choosing the 3 most important tasks, then selecting one very small first step. "
                "It is also useful to leave space for breaks and unexpected changes instead of planning every minute."
            ),
            "focus_productivity": (
                "When focus is low, it is better to reduce the size of the task instead of forcing yourself to do everything. "
                "Choose one clear task, remove distractions for a short period and start with a small action."
            ),
            "break_management": (
                "Breaks should help you return to work with a clearer mind. "
                "After a short coffee or tea break, choose one small task and continue with a simple work block."
            ),
            "daily_routine": (
                "A daily routine should be simple and repeatable. "
                "A short morning start, a few realistic tasks and a small evening review can make the day feel more organized."
            ),
            "simple_meals": (
                "Simple meal planning should focus on easy and manageable options. "
                "Instead of a strict diet plan, it can help to choose practical meals that fit your daily routine."
            ),
        }

        answer = templates.get(
            topic,
            "I found related information in my knowledge base, but I could not create a detailed answer from the selected text."
        )

        return f"{answer}\n\nSource: {source_file}"

    templates = {
        "daily_planning": (
            "Günlük planlama, günü tamamen doldurmak için değil günü daha yönetilebilir hale getirmek için yapılmalıdır. "
            "Pratik bir plan için önce günün en önemli 3 görevi seçilebilir, ardından bu görevlerden birine çok küçük bir başlangıç adımı belirlenebilir. "
            "Planın içine mola ve esneklik payı bırakmak da önemlidir; çünkü her dakikayı dolduran planlar çoğu zaman sürdürülebilir olmaz."
        ),
        "focus_productivity": (
            "Odaklanmakta zorlanıyorsan önce görevi küçültmek daha doğru olur. "
            "Tek bir net görev seçip kısa bir süre için dikkat dağıtıcı şeyleri azaltabilir ve işe çok küçük bir adımla başlayabilirsin."
        ),
        "break_management": (
            "Mola, çalışmayı bölmekten çok zihni toparlamak için kullanılmalıdır. "
            "Kahve veya çay molasından sonra yeniden başlamak için tek bir küçük görev seçmek ve kısa bir çalışma bloğu belirlemek faydalı olur."
        ),
        "daily_routine": (
            "Günlük rutin sade ve tekrar edilebilir olmalıdır. "
            "Kısa bir sabah başlangıcı, gerçekçi birkaç görev ve gün sonunda küçük bir değerlendirme, günü daha düzenli hissettirebilir."
        ),
        "simple_meals": (
            "Basit yemek planı, karmaşık bir diyet listesi gibi değil günlük rutine uyacak pratik seçenekler gibi düşünülmelidir. "
            "Çok uğraştırmayan, kolay hazırlanabilen ve günü aksatmayan öğünler tercih edilebilir."
        ),
        "safe_fallback": (
            "Bu konu botun ana bilgi alanı dışında veya dikkatli cevap verilmesi gereken bir alan olabilir. "
            "Kesin karar veya profesyonel tavsiye vermek yerine güvenilir kaynaklara ya da ilgili uzmana danışmak daha doğru olur."
        ),
    }

    answer = templates.get(
        topic,
        "Bilgi tabanında ilgili bir kaynak buldum ancak seçilen metinden net bir cevap oluşturamadım."
    )

    return f"{answer}\n\nKaynak: {source_file}"


def build_english_direct_answer(question, rag_results):
    source_file = rag_results[0].get("source_file", "")

    sentences = extract_relevant_sentences(
        question=question,
        rag_results=rag_results,
        max_sentences=4
    )

    if not sentences:
        return build_topic_template_answer(
            question=question,
            source_file=source_file,
            language="en"
        )

    answer = "According to the relevant parts of my knowledge base, "

    clean_sentences = []

    for sentence in sentences:
        if has_bad_user_visible_marker(sentence) or is_help_section(sentence):
            continue

        clean_sentences.append(sentence)

    if not clean_sentences:
        return build_topic_template_answer(
            question=question,
            source_file=source_file,
            language="en"
        )

    answer += " ".join(clean_sentences[:3])

    return f"{answer}\n\nSource: {source_file}"


def build_turkish_direct_answer(question, rag_results):
    source_file = rag_results[0].get("source_file", "")

    topic = infer_query_topic(question)

    if topic in [
        "daily_planning",
        "focus_productivity",
        "break_management",
        "daily_routine",
        "simple_meals",
        "safe_fallback"
    ]:
        return build_topic_template_answer(
            question=question,
            source_file=source_file,
            language="tr"
        )

    sentences = extract_relevant_sentences(
        question=question,
        rag_results=rag_results,
        max_sentences=4
    )

    clean_sentences = []

    for sentence in sentences:
        if has_bad_user_visible_marker(sentence) or is_help_section(sentence):
            continue

        clean_sentences.append(sentence)

    if not clean_sentences:
        return build_topic_template_answer(
            question=question,
            source_file=source_file,
            language="tr"
        )

    answer = "Bilgi tabanındaki ilgili bölümlere göre, "
    answer += " ".join(clean_sentences[:3])

    return f"{answer}\n\nKaynak: {source_file}"


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

    if detected_language == "en":
        return build_english_direct_answer(question, rag_results)

    return build_turkish_direct_answer(question, rag_results)