import json
import os
import re
from datetime import datetime


DOCUMENTS_DIR = "documents"
OUTPUT_INDEX_FILE = "knowledge_index.json"

MAX_CHUNK_CHARS = 1400
MIN_CHUNK_CHARS = 300


TOPIC_KEYWORDS = {
    "daily_planning": [
        "plan", "planning", "planner", "agenda", "task", "tasks",
        "planlama", "ajanda", "görev", "günlük plan", "öncelik",
        "yapılacaklar", "plan defteri"
    ],
    "focus_productivity": [
        "focus", "productivity", "productive", "motivation",
        "odak", "verimlilik", "verimli", "motivasyon", "dikkat",
        "çalışma düzeni", "başlamak"
    ],
    "break_management": [
        "break", "coffee", "tea", "rest",
        "mola", "kahve", "çay", "dinlenme", "kısa ara",
        "çalışma bloğu"
    ],
    "daily_routine": [
        "routine", "morning", "habit",
        "rutin", "sabah", "alışkanlık", "günlük",
        "akşam rutini"
    ],
    "simple_meals": [
        "meal", "food", "eat", "diet", "healthy",
        "yemek", "beslenme", "basit yemek", "sağlıklı",
        "pratik öğün"
    ],
    "weather_preparation": [
        "weather", "rain", "umbrella", "cold", "hot",
        "hava", "yağmur", "şemsiye", "soğuk", "sıcak",
        "mont", "ceket", "rüzgar"
    ],
    "session_memory": [
        "session", "history", "memory", "previous message",
        "konuşma geçmişi", "az önce", "ona göre", "devam et",
        "önceki mesaj", "timer", "5 dakika"
    ],
    "api_security": [
        "api key", "jwt", "middleware", "authorization",
        "x-api-key", "endpoint", "token", "api güvenliği",
        "erişim kontrolü"
    ],
    "safe_fallback": [
        "fallback", "safe", "out of scope", "investment",
        "bitcoin", "medical", "legal", "kapsam dışı",
        "yatırım tavsiyesi", "sağlık", "hukuk", "güvenli cevap"
    ],
    "evaluation": [
        "evaluation", "test", "threshold", "top 3",
        "değerlendirme", "test seti", "başarı kriteri",
        "rag log", "tool log"
    ]
}


def normalize_text(text):
    text = str(text or "").lower()
    text = text.replace("ı", "i")
    return text


def tokenize(text):
    normalized = normalize_text(text)
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", normalized)

    return [
        token.lower().strip()
        for token in tokens
        if len(token.strip()) > 2
    ]


def clean_extracted_text(text):
    text = str(text or "")

    # PDF header/footer temizliği
    text = re.sub(r"Sayfa\s*/\s*Page\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s+\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"RAG knowledge document\s*-\s*generated.*", " ", text, flags=re.IGNORECASE)

    # Çoklu boşluk temizliği
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_metadata_or_toc_text(text):
    lowered = normalize_text(text)

    topic_count = lowered.count("topic:")
    keyword_count = lowered.count("keywords:")

    metadata_markers = [
        "konu / topic",
        "anahtar kelimeler / keywords",
        "bu doküman, rag sistemi",
        "this document is prepared as",
        "generated for telegram ai daily bot"
    ]

    if topic_count >= 4:
        return True

    if keyword_count >= 4:
        return True

    if any(marker in lowered for marker in metadata_markers):
        return True

    return False


def read_txt_or_md(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def read_docx(file_path):
    try:
        from docx import Document
    except ImportError:
        raise ImportError("DOCX okumak için python-docx kurmalısın: pip install python-docx")

    document = Document(file_path)
    paragraphs = []

    for paragraph in document.paragraphs:
        paragraph_text = paragraph.text.strip()

        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def read_pdf(file_path):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("PDF okumak için pypdf kurmalısın: pip install pypdf")

    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        page_text = clean_extracted_text(page_text)

        if page_text.strip():
            pages.append(page_text)

    return "\n\n".join(pages)


def read_document(file_path):
    extension = os.path.splitext(file_path)[1].lower()

    if extension in [".txt", ".md"]:
        return clean_extracted_text(read_txt_or_md(file_path))

    if extension == ".docx":
        return clean_extracted_text(read_docx(file_path))

    if extension == ".pdf":
        return clean_extracted_text(read_pdf(file_path))

    raise ValueError(f"Desteklenmeyen dosya tipi: {extension}")


def split_into_paragraphs(text):
    parts = re.split(r"\n\s*\n", text)
    paragraphs = []

    for part in parts:
        clean = re.sub(r"\s+", " ", part).strip()

        if clean:
            paragraphs.append(clean)

    return paragraphs


def chunk_document(text, max_chars=MAX_CHUNK_CHARS, min_chars=MIN_CHUNK_CHARS):
    paragraphs = split_into_paragraphs(text)

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if is_metadata_or_toc_text(paragraph):
            continue

        if not current_chunk:
            current_chunk = paragraph
            continue

        candidate = current_chunk + "\n\n" + paragraph

        if len(candidate) <= max_chars:
            current_chunk = candidate
        else:
            if len(current_chunk) >= min_chars and not is_metadata_or_toc_text(current_chunk):
                chunks.append(current_chunk)

            current_chunk = paragraph

    if current_chunk.strip() and not is_metadata_or_toc_text(current_chunk):
        chunks.append(current_chunk)

    clean_chunks = []

    for chunk in chunks:
        chunk = clean_extracted_text(chunk)

        if len(chunk) < min_chars:
            continue

        if is_metadata_or_toc_text(chunk):
            continue

        clean_chunks.append(chunk)

    return clean_chunks


def infer_topic(text):
    normalized = normalize_text(text)

    topic_scores = {}

    for topic, keywords in TOPIC_KEYWORDS.items():
        score = 0

        for keyword in keywords:
            if normalize_text(keyword) in normalized:
                score += 1

        topic_scores[topic] = score

    best_topic = max(topic_scores, key=topic_scores.get)

    if topic_scores[best_topic] == 0:
        return "general"

    return best_topic


def extract_keywords(text, limit=20):
    tokens = tokenize(text)

    frequency = {}

    stopwords = {
        "bir", "ve", "veya", "ile", "için", "olan", "olarak", "daha", "çok",
        "ama", "fakat", "gibi", "kadar", "sonra", "önce",
        "the", "and", "for", "with", "that", "this", "from", "you", "your",
        "can", "should", "would", "about"
    }

    for token in tokens:
        if token in stopwords:
            continue

        frequency[token] = frequency.get(token, 0) + 1

    sorted_items = sorted(
        frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [item[0] for item in sorted_items[:limit]]


def build_knowledge_index(
    documents_dir=DOCUMENTS_DIR,
    output_file=OUTPUT_INDEX_FILE
):
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)

    index = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "documents_dir": documents_dir,
        "chunk_count": 0,
        "chunks": []
    }

    supported_extensions = [".txt", ".md", ".docx", ".pdf"]

    for file_name in os.listdir(documents_dir):
        file_path = os.path.join(documents_dir, file_name)
        extension = os.path.splitext(file_name)[1].lower()

        if not os.path.isfile(file_path):
            continue

        if extension not in supported_extensions:
            continue

        try:
            content = read_document(file_path)
            chunks = chunk_document(content)

            for chunk_index, chunk_text in enumerate(chunks, start=1):
                chunk_id = f"{file_name}-{chunk_index}"

                index["chunks"].append({
                    "chunk_id": chunk_id,
                    "source_file": file_name,
                    "topic": infer_topic(chunk_text),
                    "keywords": extract_keywords(chunk_text),
                    "text": chunk_text
                })

            print(f"İşlendi: {file_name} - {len(chunks)} chunk")

        except Exception as error:
            print(f"Doküman işlenemedi: {file_name} - {error}")

    index["chunk_count"] = len(index["chunks"])

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(index, file, ensure_ascii=False, indent=2)

    print(f"Knowledge index oluşturuldu: {output_file}")
    print(f"Chunk sayısı: {index['chunk_count']}")


if __name__ == "__main__":
    build_knowledge_index()