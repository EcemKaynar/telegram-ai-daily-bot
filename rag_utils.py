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

def get_combined_question_context(question, history_context=""):
    combined = f"{history_context or ''}\n\nKullanıcının son mesajı: {question or ''}"
    return combined.strip()


def extract_time_hint(text):
    normalized = normalize_text(text)

    hour_patterns = [
        r"(\d+)\s*saat",
        r"(\d+)\s*hour"
    ]

    for pattern in hour_patterns:
        match = re.search(pattern, normalized)

        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None

    if "yarim saat" in normalized or "yarım saat" in normalized:
        return 0.5

    return None


def has_any(text, keywords):
    normalized = normalize_text(text)

    return any(normalize_text(keyword) in normalized for keyword in keywords)


def build_morning_routine_answer(question, source_file, history_context=""):
    combined = get_combined_question_context(question, history_context)
    available_hours = extract_time_hint(combined)

    likes_walking = has_any(combined, ["yürüyüş", "yuruyus", "walk", "walking"])
    likes_outside = has_any(combined, ["dışarı", "disari", "outside"])
    likes_social = has_any(combined, ["arkadaş", "arkadas", "sohbet", "friend", "friends", "chat"])

    if available_hours and available_hours >= 3:
        routine = (
            "Tabii, verdiğin bilgilere göre 3 saatlik daha keyifli ve sürdürülebilir bir sabah rutini şöyle olabilir:\n\n"
            "1. İlk 15 dakika: Güne yavaş başla, su iç, yüzünü yıka ve kendini toparla.\n"
            "2. 15-30 dakika: Hafif bir kahvaltı veya kahve/çay hazırlığı yap.\n"
            "3. 45-60 dakika: Dışarı çıkıp yürüyüş yap. Bu kısmı hem hareket hem de zihni açma zamanı gibi düşünebilirsin.\n"
            "4. 30-45 dakika: Eve dönünce duş, hazırlanma veya kısa toparlanma zamanı ayır.\n"
            "5. 30 dakika: Günün en önemli 2-3 görevini yaz. Çok uzun liste yapmadan sadece öncelikleri belirle.\n"
            "6. Kalan zaman: Arkadaşlarınla kısa bir sohbet, mesajlaşma veya keyifli bir mola ekleyebilirsin.\n\n"
            "Bu rutinin amacı sabahı sadece görevlerle doldurmak değil; hareket, hazırlık, planlama ve sosyal keyfi dengeli şekilde birleştirmek."
        )

        return f"{routine}\n\nKaynak: {source_file}"

    if likes_walking or likes_outside or likes_social:
        routine = (
            "Tabii, sevdiğin şeylere göre daha doğal bir sabah rutini oluşturabiliriz:\n\n"
            "1. Güne su içip kendini toparlayarak başla.\n"
            "2. Kısa bir kahvaltı veya kahve/çay molası koy.\n"
            "3. Dışarı çıkıp yürüyüş yapabileceğin bir zaman ayır.\n"
            "4. Eve dönünce günün en önemli 2-3 görevini yaz.\n"
            "5. Arkadaşlarınla sohbet etmek istiyorsan bunu rutinin ödül veya keyif kısmı gibi ekleyebilirsin.\n\n"
            "Böylece sabah rutinin hem düzenli hem de sana iyi gelen şeyleri içeren bir yapıya dönüşür."
        )

        return f"{routine}\n\nKaynak: {source_file}"

    routine = (
        "Tabii, senin için sade ve uygulanabilir bir sabah rutini şöyle olabilir:\n\n"
        "1. İlk 5-10 dakika: Su iç, yüzünü yıka ve kendini toparla.\n"
        "2. 10-20 dakika: Hafif kahvaltı veya kahve/çay hazırlığı yap.\n"
        "3. 10 dakika: Bugünün ana hedefini ve en önemli 3 görevini yaz.\n"
        "4. 25-30 dakika: İlk küçük göreve başla veya kısa bir yürüyüş yap.\n"
        "5. Son 5 dakika: Gün içinde ihtiyacın olacak şeyleri kontrol et.\n\n"
        "Bu rutini çok uzun tutmadan başlatmak daha iyi olur. Birkaç gün denedikten sonra sana iyi gelen kısımları koruyup gereksiz gelenleri çıkarabilirsin."
    )

    return f"{routine}\n\nKaynak: {source_file}"


def build_daily_planning_answer(question, source_file, history_context=""):
    answer = (
        "Tabii, senin için uygulanabilir bir günlük plan yapısı şöyle olabilir:\n\n"
        "1. Önce aklındaki tüm işleri kısa bir listeye dök.\n"
        "2. Bu listeden bugün gerçekten önemli olan 3 görevi seç.\n"
        "3. Her görev için sadece ilk küçük adımı belirle.\n"
        "4. Günü 2-3 çalışma bloğuna ayır ve aralara kısa molalar koy.\n"
        "5. Planın tamamını doldurma; beklenmeyen işler için boşluk bırak.\n"
        "6. Gün sonunda neyin işe yaradığını ve yarına ne kalacağını kısaca kontrol et.\n\n"
        "Örnek mini plan:\n"
        "- Sabah: En önemli görevin ilk adımı\n"
        "- Öğlen: Daha kısa işler veya iletişim işleri\n"
        "- Öğleden sonra: İkinci odak bloğu\n"
        "- Akşam: Kısa toparlama ve yarının ana hedefini belirleme"
    )

    return f"{answer}\n\nKaynak: {source_file}"


def build_focus_answer(question, source_file, history_context=""):
    answer = (
        "Odaklanmakta zorlanıyorsan önce işi büyütmeden başlamak daha iyi olur:\n\n"
        "1. Şu an yapman gereken tek bir görevi seç.\n"
        "2. Bu görevi 5 dakikada başlayabileceğin kadar küçült.\n"
        "3. Telefon, bildirim veya açık sekmeler gibi dikkat dağıtıcıları kısa süreliğine azalt.\n"
        "4. 10-15 dakikalık küçük bir çalışma bloğu başlat.\n"
        "5. Blok bitince kısa mola ver ve devam edip etmeyeceğine karar ver.\n\n"
        "Amaç bir anda çok verimli olmak değil; yeniden başlamayı kolaylaştırmak."
    )

    return f"{answer}\n\nKaynak: {source_file}"


def build_break_answer(question, source_file, history_context=""):
    answer = (
        "Mola sonrası çalışmaya dönmek için molaya çıkmadan önce dönüş adımını belirlemek iyi olur:\n\n"
        "1. Moladan önce döndüğünde yapacağın ilk küçük işi yaz.\n"
        "2. Mola süresini net belirle; örneğin 10 dakika.\n"
        "3. Molada gerçekten zihnini toparlayacak bir şey yap: su içmek, kısa yürüyüş, esneme gibi.\n"
        "4. Mola bitince sadece yazdığın küçük adıma başla.\n"
        "5. Devam etmek kolay gelirse çalışma bloğunu uzatabilirsin.\n\n"
        "Örneğin 'çalışmaya döneceğim' yerine 'raporun ilk paragrafını düzenleyeceğim' demek daha işe yarar."
    )

    return f"{answer}\n\nKaynak: {source_file}"


def build_meal_answer(question, source_file, history_context=""):
    answer = (
        "Bugün basit ve uğraştırmayan bir şey istiyorsan şu seçeneklerden birini seçebilirsin:\n\n"
        "1. Yumurtalı tost + ayran veya çay\n"
        "2. Yoğurt + yulaf + meyve\n"
        "3. Makarna + yoğurt + salata\n"
        "4. Omlet + ekmek + domates/salatalık\n"
        "5. Mercimek çorbası + ekmek\n\n"
        "Hafif bir şey istiyorsan yoğurtlu seçenek daha iyi olur. Daha doyurucu bir şey istiyorsan tost, omlet veya makarna daha uygun olabilir."
    )

    return f"{answer}\n\nKaynak: {source_file}"


def build_safety_answer(question, source_file, history_context=""):
    answer = (
        "Bu konu karar verirken dikkatli olunması gereken bir alan gibi görünüyor. "
        "Doğrudan yatırım, sağlık, hukuk veya benzeri profesyonel karar tavsiyesi vermem doğru olmaz.\n\n"
        "Daha güvenli yaklaşım şu olabilir:\n"
        "1. Kararı tek bir cevaba göre verme.\n"
        "2. Güvenilir kaynaklardan araştır.\n"
        "3. Riskleri ve kendi durumunu değerlendir.\n"
        "4. Gerekirse ilgili uzmana danış.\n\n"
        "İstersen bu konuyu karar verirken dikkat edilecek genel başlıklar şeklinde toparlayabilirim."
    )

    return f"{answer}\n\nKaynak: {source_file}"
def build_topic_template_answer(question, source_file, language, history_context=""):
    topic = infer_query_topic(question)

    combined_context = get_combined_question_context(question, history_context)
    normalized_combined = normalize_text(combined_context)

    if language == "en":
        templates = {
            "daily_planning": (
                "A practical daily plan can start with choosing the 3 most important tasks, "
                "breaking each one into a small first step, and leaving space for breaks and unexpected changes."
            ),
            "focus_productivity": (
                "When focus is low, choose one clear task, reduce it to a small first step, "
                "remove distractions for a short period and start with a short work block."
            ),
            "break_management": (
                "A good break should make it easier to return to work. Before the break, decide the first small step you will do when you come back."
            ),
            "daily_routine": (
                "A useful routine should be simple, repeatable and realistic. Start with a short morning structure, a few clear tasks and a small review."
            ),
            "simple_meals": (
                "For a simple meal, choose an easy option such as eggs and toast, yogurt with oats and fruit, pasta with yogurt, or a quick salad."
            ),
            "safe_fallback": (
                "This topic may require professional judgment. I cannot give a direct decision, but I can help you think through general safety points."
            ),
        }

        answer = templates.get(
            topic,
            "I found related information in my knowledge base, but I could not create a detailed answer from the selected text."
        )

        return f"{answer}\n\nSource: {source_file}"

    if topic == "daily_routine":
        if "sabah" in normalized_combined or "morning" in normalized_combined or "rutin" in normalized_combined:
            return build_morning_routine_answer(
                question=question,
                source_file=source_file,
                history_context=history_context
            )

    if topic == "daily_planning":
        return build_daily_planning_answer(
            question=question,
            source_file=source_file,
            history_context=history_context
        )

    if topic == "focus_productivity":
        return build_focus_answer(
            question=question,
            source_file=source_file,
            history_context=history_context
        )

    if topic == "break_management":
        return build_break_answer(
            question=question,
            source_file=source_file,
            history_context=history_context
        )

    if topic == "simple_meals":
        return build_meal_answer(
            question=question,
            source_file=source_file,
            history_context=history_context
        )

    if topic == "safe_fallback":
        return build_safety_answer(
            question=question,
            source_file=source_file,
            history_context=history_context
        )

    answer = (
        "Bilgi tabanında bu konuyla ilgili kaynak buldum. "
        "Buna göre daha net yardımcı olabilmem için isteğini biraz daha spesifik yazabilirsin. "
        "Örneğin plan, rutin, odaklanma, mola veya yemek önerisi istediğini belirtebilirsin."
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

def build_turkish_direct_answer(question, rag_results, history_context=""):
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
            language="tr",
            history_context=history_context
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
            language="tr",
            history_context=history_context
        )

    answer = "Bilgi tabanındaki ilgili bölümlere göre, "
    answer += " ".join(clean_sentences[:3])

    return f"{answer}\n\nKaynak: {source_file}"

def build_direct_rag_answer(question, rag_results, language=None, history_context=""):
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
            "Şu an günlük planlama, odaklanma, mola yönetimi, rutin oluşturma, "
            "basit yemek fikirleri ve hava durumuna göre hazırlık konularında daha iyi yardımcı olabilirim."
        )

    if detected_language == "en":
        return build_english_direct_answer(question, rag_results)

    return build_turkish_direct_answer(
        question=question,
        rag_results=rag_results,
        history_context=history_context
    )