import argparse
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

from qdrant_client import QdrantClient, models


DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "daily_assistant_kb")
QDRANT_EMBEDDING_MODEL = os.getenv("QDRANT_EMBEDDING_MODEL", "BAAI/bge-small-en")
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
QDRANT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.35"))

MAX_CHUNK_CHARS = int(os.getenv("QDRANT_MAX_CHUNK_CHARS", "1600"))
MIN_CHUNK_CHARS = int(os.getenv("QDRANT_MIN_CHUNK_CHARS", "300"))

DOCUMENT_NAMESPACE = uuid.UUID("12345678-1234-5678-1234-567812345678")


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


def refresh_env_values():
    global DOCUMENTS_DIR
    global QDRANT_MODE
    global QDRANT_PATH
    global QDRANT_URL
    global QDRANT_API_KEY
    global QDRANT_COLLECTION
    global QDRANT_EMBEDDING_MODEL
    global QDRANT_TOP_K
    global QDRANT_SCORE_THRESHOLD
    global MAX_CHUNK_CHARS
    global MIN_CHUNK_CHARS

    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "documents")
    QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "daily_assistant_kb")
    QDRANT_EMBEDDING_MODEL = os.getenv("QDRANT_EMBEDDING_MODEL", "BAAI/bge-small-en")
    QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
    QDRANT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.35"))
    MAX_CHUNK_CHARS = int(os.getenv("QDRANT_MAX_CHUNK_CHARS", "1600"))
    MIN_CHUNK_CHARS = int(os.getenv("QDRANT_MIN_CHUNK_CHARS", "300"))


def get_qdrant_client():
    if QDRANT_MODE.lower() == "cloud":
        if not QDRANT_URL:
            raise ValueError("QDRANT_MODE=cloud ama QDRANT_URL boş.")

        return QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY or None,
            timeout=60
        )

    return QdrantClient(path=QDRANT_PATH)


def normalize_space(text):
    text = str(text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_file(file_path):
    return Path(file_path).read_text(encoding="utf-8")


def clean_document_text(text):
    text = normalize_space(text)

    # Son kullanıcıya gitmemesi gereken teknik kalıntıları önlem amaçlı temizliyoruz.
    forbidden_patterns = [
        r"\bRAG_THRESHOLD\b",
        r"\bRAG_TOP_K\b",
        r"\bembedding score\b",
        r"\bchunk score\b",
        r"\bmiddleware\b",
    ]

    for pattern in forbidden_patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    return normalize_space(text)


def infer_language(file_name, text):
    lower_name = file_name.lower()

    if lower_name.endswith("_en.txt") or "_en" in lower_name:
        return "en"

    if lower_name.endswith("_tr.txt") or "_tr" in lower_name:
        return "tr"

    turkish_chars = ["ç", "ğ", "ı", "ö", "ş", "ü"]

    if any(char in text.lower() for char in turkish_chars):
        return "tr"

    return "en"

def infer_topic(file_name, section_title, text):
    lower_name = file_name.lower()

    # Önce dosya adına göre kesin topic veriyoruz.
    # Bu metadata'nın yanlış yazılmasını engeller.
    file_topic_map = {
        "01_": "assistant_operating_guide",
        "02_": "daily_planning",
        "03_": "focus_productivity",
        "04_": "habit_routine",
        "05_": "weather_preparation",
        "06_": "meal_planning",
        "07_": "safety_policy"
    }

    for prefix, topic in file_topic_map.items():
        if lower_name.startswith(prefix):
            return topic

    combined = f"{section_title} {text}".lower()

    topic_rules = {
        "meal_planning": [
            "meal",
            "food",
            "öğün",
            "ogun",
            "yemek",
            "öz bakım",
            "oz bakim",
            "daily care",
            "tost",
            "yoğurt",
            "yogurt",
            "makarna"
        ],
        "focus_productivity": [
            "focus",
            "productivity",
            "odak",
            "odaklanamıyorum",
            "odaklanamiyorum",
            "verimlilik",
            "verimsiz",
            "mola",
            "break",
            "motivation",
            "motivasyon"
        ],
        "daily_planning": [
            "planning",
            "planlama",
            "time management",
            "zaman yönetimi",
            "zaman yonetimi",
            "daily plan",
            "günlük plan",
            "gunluk plan"
        ],
        "habit_routine": [
            "habit",
            "routine",
            "alışkanlık",
            "aliskanlik",
            "rutin",
            "sabah",
            "akşam",
            "aksam",
            "morning",
            "evening"
        ],
        "weather_preparation": [
            "weather",
            "hava",
            "şemsiye",
            "semsiye",
            "umbrella",
            "rain",
            "yağmur",
            "yagmur",
            "temperature"
        ],
        "safety_policy": [
            "safety",
            "policy",
            "güvenli",
            "guvenli",
            "kapsam dışı",
            "kapsam disi",
            "finance",
            "health",
            "legal",
            "yatırım",
            "yatirim",
            "sağlık",
            "saglik",
            "hukuk",
            "bitcoin",
            "kripto"
        ],
        "assistant_operating_guide": [
            "operating guide",
            "assistant",
            "asistan",
            "kapsam",
            "cevap tarzı"
        ],
    }

    for topic, keywords in topic_rules.items():
        if any(keyword in combined for keyword in keywords):
            return topic

    return "general"


def extract_title(text, fallback):
    for line in text.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line.replace("#", "").strip()

    return fallback


def split_by_sections(text):
    lines = text.splitlines()
    sections = []

    current_title = "General"
    current_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("## "):
            if current_lines:
                sections.append({
                    "section_title": current_title,
                    "text": "\n".join(current_lines).strip()
                })

            current_title = stripped.replace("#", "").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "section_title": current_title,
            "text": "\n".join(current_lines).strip()
        })

    return sections


def split_long_text(text, max_chars):
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)

            for sentence in sentences:
                sentence = sentence.strip()

                if not sentence:
                    continue

                candidate = f"{current}\n\n{sentence}".strip()

                if len(candidate) <= max_chars:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)

                    current = sentence

            continue

        candidate = f"{current}\n\n{paragraph}".strip()

        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)

            current = paragraph

    if current:
        chunks.append(current)

    return chunks


def build_chunks_for_file(file_path):
    file_path = Path(file_path)
    source_file = file_path.name
    raw_text = read_text_file(file_path)
    text = clean_document_text(raw_text)

    if not text:
        return []

    document_title = extract_title(text, fallback=source_file)
    language = infer_language(source_file, text)
    document_id = source_file.replace(".txt", "")
    sections = split_by_sections(text)

    chunks = []
    chunk_index = 0

    for section in sections:
        section_title = section["section_title"]
        section_text = section["text"]

        section_chunks = split_long_text(
            text=section_text,
            max_chars=MAX_CHUNK_CHARS
        )

        for chunk_text in section_chunks:
            chunk_text = normalize_space(chunk_text)

            if len(chunk_text) < MIN_CHUNK_CHARS:
                continue

            topic = infer_topic(
                file_name=source_file,
                section_title=section_title,
                text=chunk_text
            )

            chunk_id = f"{document_id}_chunk_{chunk_index:04d}"
            point_id = str(uuid.uuid5(DOCUMENT_NAMESPACE, chunk_id))

            metadata = {
                "document_id": document_id,
                "source_file": source_file,
                "title": document_title,
                "language": language,
                "topic": topic,
                "section_title": section_title,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "document": chunk_text,
                "created_at": datetime.now().strftime("%Y-%m-%d"),
                "version": "1.0"
            }

            chunks.append({
                "id": point_id,
                "text": chunk_text,
                "metadata": metadata
            })

            chunk_index += 1

    return chunks


def load_all_document_chunks():
    documents_path = Path(DOCUMENTS_DIR)

    if not documents_path.exists():
        raise FileNotFoundError(f"{DOCUMENTS_DIR} klasörü bulunamadı.")

    all_chunks = []

    txt_files = sorted(documents_path.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"{DOCUMENTS_DIR} içinde .txt dosyası bulunamadı.")

    for file_path in txt_files:
        # README veya teknik açıklama dosyasını indexlemiyoruz.
        if file_path.name.lower().startswith("readme"):
            print(f"Atlandı: {file_path.name}")
            continue

        file_chunks = build_chunks_for_file(file_path)
        all_chunks.extend(file_chunks)

        print(f"{file_path.name}: {len(file_chunks)} chunk hazırlandı.")

    return all_chunks


def collection_exists(client, collection_name):
    try:
        return client.collection_exists(collection_name)
    except Exception:
        try:
            client.get_collection(collection_name)
            return True
        except Exception:
            return False


def recreate_collection(client, collection_name):
    if collection_exists(client, collection_name):
        print(f"Eski collection siliniyor: {collection_name}")
        client.delete_collection(collection_name=collection_name)

    vector_size = client.get_embedding_size(QDRANT_EMBEDDING_MODEL)

    print(f"Collection oluşturuluyor: {collection_name}")
    print(f"Embedding model: {QDRANT_EMBEDDING_MODEL}")
    print(f"Vector size: {vector_size}")

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )


def upload_chunks_to_qdrant(client, chunks):
    if not chunks:
        raise ValueError("Yüklenecek chunk bulunamadı.")

    documents = [chunk["text"] for chunk in chunks]
    payloads = [chunk["metadata"] for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    print(f"Qdrant'a yükleniyor. Chunk sayısı: {len(chunks)}")

    client.upload_collection(
        collection_name=QDRANT_COLLECTION,
        vectors=[
            models.Document(
                text=document,
                model=QDRANT_EMBEDDING_MODEL
            )
            for document in documents
        ],
        payload=payloads,
        ids=ids
    )

    print("Qdrant upload tamamlandı.")


def test_query(client, query_text):
    print("\nTest query çalışıyor:")
    print(query_text)

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=models.Document(
            text=query_text,
            model=QDRANT_EMBEDDING_MODEL
        ),
        limit=QDRANT_TOP_K,
        score_threshold=QDRANT_SCORE_THRESHOLD
    )

    points = response.points

    print(f"\nBulunan sonuç sayısı: {len(points)}")

    for index, point in enumerate(points, start=1):
        payload = point.payload or {}

        print("\n" + "-" * 80)
        print(f"Sonuç {index}")
        print(f"Score: {round(point.score, 4)}")
        print(f"Source: {payload.get('source_file')}")
        print(f"Topic: {payload.get('topic')}")
        print(f"Section: {payload.get('section_title')}")
        print(f"Chunk ID: {payload.get('chunk_id')}")
        print("Text preview:")
        print(str(payload.get("text", ""))[:500])


def main():
    load_env_file()
    refresh_env_values()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Collection'ı silip yeniden oluşturur."
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="Upload sonrası Qdrant semantic search test sorgusu."
    )

    args = parser.parse_args()

    print("Qdrant ingest başlıyor.")
    print(f"Mode: {QDRANT_MODE}")
    print(f"Path: {QDRANT_PATH}")
    print(f"Collection: {QDRANT_COLLECTION}")
    print(f"Documents dir: {DOCUMENTS_DIR}")

    client = get_qdrant_client()

    chunks = load_all_document_chunks()

    if args.reset or not collection_exists(client, QDRANT_COLLECTION):
        recreate_collection(client, QDRANT_COLLECTION)

    upload_chunks_to_qdrant(client, chunks)

    if args.test:
        test_query(client, args.test)

    print("\nİşlem tamamlandı.")


if __name__ == "__main__":
    main()