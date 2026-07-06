import json
import os
import re
from pathlib import Path

from qdrant_client import QdrantClient, models


QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "daily_assistant_kb")
QDRANT_EMBEDDING_MODEL = os.getenv(
    "QDRANT_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
QDRANT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.0"))


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
    global QDRANT_MODE
    global QDRANT_PATH
    global QDRANT_URL
    global QDRANT_API_KEY
    global QDRANT_COLLECTION
    global QDRANT_EMBEDDING_MODEL
    global QDRANT_TOP_K
    global QDRANT_SCORE_THRESHOLD

    QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
    QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_storage")
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "daily_assistant_kb")
    QDRANT_EMBEDDING_MODEL = os.getenv(
        "QDRANT_EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    QDRANT_TOP_K = int(os.getenv("QDRANT_TOP_K", "3"))
    QDRANT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.0"))


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


def clean_text_for_prompt(text):
    text = str(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_query_text(text):
    text = str(text or "").lower()

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

    return text


def detect_query_topic(question):
    text = normalize_query_text(question)

    topic_keywords = {
        "focus_productivity": [
            "odak",
            "odaklanamiyorum",
            "dikkatim dagiliyor",
            "dikkatim daginik",
            "verimsiz",
            "verimli",
            "motivasyon",
            "motive",
            "calisamiyorum",
            "baslayamiyorum",
            "erteleme",
            "mola",
            "focus",
            "unproductive",
            "motivation",
            "concentrate",
            "can't focus",
            "cannot focus"
        ],
        "meal_planning": [
            "yemek",
            "ne yiyebilirim",
            "ne yesem",
            "ogun",
            "basit ne yiyebilirim",
            "kahvalti",
            "ogle yemegi",
            "aksam yemegi",
            "meal",
            "food",
            "eat",
            "breakfast",
            "lunch",
            "dinner"
        ],
        "safety_policy": [
            "bitcoin",
            "coin",
            "kripto",
            "crypto",
            "hisse",
            "borsa",
            "yatirim",
            "yatirim tavsiyesi",
            "alinir mi",
            "satilir mi",
            "doktor",
            "ilac",
            "recete",
            "hukuk",
            "avukat",
            "investment",
            "medical",
            "legal"
        ],
        "weather_preparation": [
            "hava",
            "hava durumu",
            "yagmur",
            "semsiye",
            "sicaklik",
            "ruzgar",
            "kar",
            "mont",
            "ceket",
            "weather",
            "rain",
            "umbrella",
            "temperature",
            "wind",
            "snow"
        ],
        "habit_routine": [
            "rutin",
            "sabah rutini",
            "aksam rutini",
            "aliskanlik",
            "habit",
            "morning routine",
            "evening routine"
        ],
        "daily_planning": [
            "gunluk plan",
            "gunluk planlama",
            "planlama",
            "plan yap",
            "program yap",
            "zaman yonetimi",
            "daily plan",
            "daily planning",
            "time management",
            "schedule"
        ],
    }

    for topic, keywords in topic_keywords.items():
        if any(keyword in text for keyword in keywords):
            return topic

    return "general"


def rerank_qdrant_results(question, results, final_top_k):
    query_topic = detect_query_topic(question)

    reranked = []

    for result in results:
        original_score = float(result.get("score", 0))
        adjusted_score = original_score

        result_topic = result.get("topic", "")
        source_file = result.get("source_file", "")

        if query_topic != "general":
            if result_topic == query_topic:
                adjusted_score += 0.35
            else:
                adjusted_score -= 0.08

        # Dosya adına göre ekstra güvenli boost.
        if query_topic == "assistant_operating_guide" and source_file.startswith("01_"):
            adjusted_score += 0.25

        if query_topic == "daily_planning" and source_file.startswith("02_"):
            adjusted_score += 0.25

        if query_topic == "focus_productivity" and source_file.startswith("03_"):
            adjusted_score += 0.25

        if query_topic == "habit_routine" and source_file.startswith("04_"):
            adjusted_score += 0.25

        if query_topic == "weather_preparation" and source_file.startswith("05_"):
            adjusted_score += 0.25

        if query_topic == "meal_planning" and source_file.startswith("06_"):
            adjusted_score += 0.25

        if query_topic == "safety_policy" and source_file.startswith("07_"):
            adjusted_score += 0.25

        result["original_score"] = original_score
        result["score"] = adjusted_score
        result["query_topic"] = query_topic

        reranked.append(result)

    reranked.sort(key=lambda item: item.get("score", 0), reverse=True)

    return reranked[:final_top_k]


def search_qdrant_knowledge_base(question, top_k=None, score_threshold=None):
    load_env_file()
    refresh_env_values()

    if top_k is None:
        top_k = QDRANT_TOP_K

    if score_threshold is None:
        score_threshold = QDRANT_SCORE_THRESHOLD

    client = get_qdrant_client()

    try:
        # Önce top_k'dan daha fazla aday çekiyoruz.
        # Sonra metadata/topic bazlı rerank yapıyoruz.
        candidate_limit = max(top_k * 3, 8)

        query_kwargs = {
            "collection_name": QDRANT_COLLECTION,
            "query": models.Document(
                text=question,
                model=QDRANT_EMBEDDING_MODEL
            ),
            "limit": candidate_limit
        }

        # Threshold 0 veya daha küçükse filtrelemiyoruz.
        # Çünkü bazı doğru sonuçlar düşük skorla gelebiliyor.
        if score_threshold and score_threshold > 0:
            query_kwargs["score_threshold"] = score_threshold

        response = client.query_points(**query_kwargs)

        results = []

        for point in response.points:
            payload = point.payload or {}

            result = {
                "score": float(point.score),
                "original_score": float(point.score),
                "source_file": payload.get("source_file", ""),
                "document_id": payload.get("document_id", ""),
                "title": payload.get("title", ""),
                "language": payload.get("language", ""),
                "topic": payload.get("topic", ""),
                "section_title": payload.get("section_title", ""),
                "chunk_id": payload.get("chunk_id", ""),
                "chunk_index": payload.get("chunk_index", ""),
                "text": payload.get("text", "")
            }

            results.append(result)

        final_results = rerank_qdrant_results(
            question=question,
            results=results,
            final_top_k=top_k
        )

        return {
            "found": len(final_results) > 0,
            "results": final_results,
            "top_k": top_k,
            "score_threshold": score_threshold,
            "retriever": "qdrant",
            "query_topic": detect_query_topic(question)
        }

    except Exception as error:
        print(f"Qdrant search hatası: {error}")

        return {
            "found": False,
            "results": [],
            "top_k": top_k,
            "score_threshold": score_threshold,
            "retriever": "qdrant",
            "query_topic": detect_query_topic(question),
            "error": str(error)
        }


def format_qdrant_rag_context(results):
    context_parts = []

    for result in results:
        clean_text = clean_text_for_prompt(result.get("text", ""))

        context_parts.append(
            f"Source file: {result.get('source_file', '')}\n"
            f"Title: {result.get('title', '')}\n"
            f"Topic: {result.get('topic', '')}\n"
            f"Section: {result.get('section_title', '')}\n"
            f"Original score: {round(float(result.get('original_score', 0)), 4)}\n"
            f"Reranked score: {round(float(result.get('score', 0)), 4)}\n"
            f"Content:\n{clean_text}"
        )

    return "\n\n---\n\n".join(context_parts)


def get_qdrant_sources_json(results):
    sources = []

    for result in results:
        sources.append({
            "retriever": "qdrant",
            "source_file": result.get("source_file", ""),
            "document_id": result.get("document_id", ""),
            "topic": result.get("topic", ""),
            "section_title": result.get("section_title", ""),
            "chunk_id": result.get("chunk_id", ""),
            "original_score": result.get("original_score", 0),
            "reranked_score": result.get("score", 0),
            "query_topic": result.get("query_topic", "")
        })

    return json.dumps(sources, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]).strip()

    if not query:
        query = "Günlük planlama nasıl yapılır?"

    search_result = search_qdrant_knowledge_base(query)

    print("\nQuery:")
    print(query)

    print("\nDetected topic:")
    print(search_result.get("query_topic", ""))

    print("\nFound:")
    print(search_result["found"])

    print("\nResults:")

    for index, result in enumerate(search_result["results"], start=1):
        print("\n" + "-" * 80)
        print(f"Result {index}")
        print(f"Original score: {round(float(result.get('original_score', 0)), 4)}")
        print(f"Reranked score: {round(float(result.get('score', 0)), 4)}")
        print(f"Source: {result.get('source_file')}")
        print(f"Topic: {result.get('topic')}")
        print(f"Section: {result.get('section_title')}")
        print(f"Chunk ID: {result.get('chunk_id')}")
        print("Preview:")
        print(str(result.get("text", ""))[:500])