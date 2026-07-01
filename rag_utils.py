import os
import re
import json


KNOWLEDGE_BASE_DIR = "knowledge_base"


STOPWORDS = {
    "ve", "veya", "ile", "için", "ama", "fakat", "bir", "bu", "şu", "o",
    "ben", "sen", "bana", "beni", "ne", "nasıl", "neden", "mi", "mı",
    "mu", "mü", "de", "da", "ki", "çok", "az", "gibi", "şey", "şeyi",
    "bugün", "yarın"
}


def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ0-9]+", text)

    return [
        token
        for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]


def expand_query_tokens(tokens):
    expanded = set(tokens)

    expansions = {
        "verimsiz": ["odak", "başlamak", "görev", "motivasyon"],
        "hissetmiyorum": ["verimsiz", "odak", "başlamak", "görev"],
        "isteksiz": ["odak", "başlamak", "görev", "motivasyon"],
        "motivasyon": ["başlamak", "görev", "odak"],
        "plan": ["planlama", "görev", "ajanda"],
        "planlama": ["plan", "görev", "ajanda"],
        "defter": ["ajanda", "planlama", "yapılacaklar"],
        "ajanda": ["plan", "planlama", "görev"],
        "liste": ["yapılacaklar", "görev", "plan"],
        "kahve": ["mola", "odak", "çalışma"],
        "çay": ["mola", "odak", "çalışma"],
        "mola": ["dinlenme", "odak", "çalışma"],
        "masa": ["çalışma", "ortam", "düzen"],
        "çalışma": ["odak", "masa", "görev"],
        "yemek": ["öğün", "pratik", "rutin"],
        "rutin": ["günlük", "alışkanlık", "planlama"],
        "sabah": ["rutin", "günlük", "alışkanlık"],
        "hava": ["şemsiye", "kıyafet", "hazırlık"],
        "şemsiye": ["hava", "yağmur", "hazırlık"]
    }

    for token in tokens:
        if token in expansions:
            expanded.update(expansions[token])

    return list(expanded)


def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_document_into_chunks(file_name, content):
    chunks = []
    current_title = file_name
    paragraphs = []

    lines = content.splitlines()

    for line in lines:
        clean_line = line.strip()

        if not clean_line:
            continue

        if clean_line.startswith("#"):
            current_title = clean_line.replace("#", "").strip()
            continue

        paragraphs.append(clean_line)

    for index, paragraph in enumerate(paragraphs):
        chunks.append(
            {
                "source_file": file_name,
                "source_title": current_title,
                "chunk_id": f"{file_name}-{index + 1}",
                "text": paragraph,
                "tokens": tokenize(paragraph)
            }
        )

    return chunks


def load_knowledge_base():
    chunks = []

    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return chunks

    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
        content = read_text_file(file_path)

        document_chunks = split_document_into_chunks(
            file_name=file_name,
            content=content
        )

        chunks.extend(document_chunks)

    return chunks


def calculate_score(query_tokens, chunk_tokens, chunk_text):
    if not query_tokens or not chunk_tokens:
        return 0

    score = 0
    chunk_token_set = set(chunk_tokens)
    chunk_text_lower = chunk_text.lower()

    for token in query_tokens:
        if token in chunk_token_set:
            score += 2
        elif token in chunk_text_lower:
            score += 1

    return score


def search_knowledge_base(question, max_results=3, min_score=2):
    chunks = load_knowledge_base()

    original_tokens = tokenize(question)
    query_tokens = expand_query_tokens(original_tokens)

    scored_chunks = []

    for chunk in chunks:
        score = calculate_score(
            query_tokens=query_tokens,
            chunk_tokens=chunk["tokens"],
            chunk_text=chunk["text"]
        )

        if score >= min_score:
            scored_chunks.append(
                {
                    "score": score,
                    "source_file": chunk["source_file"],
                    "source_title": chunk["source_title"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"]
                }
            )

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    results = scored_chunks[:max_results]

    return {
        "found": len(results) > 0,
        "results": results,
        "query_tokens": query_tokens
    }


def format_rag_context(rag_results):
    context_parts = []

    for index, item in enumerate(rag_results, start=1):
        context_parts.append(
            f"Kaynak {index}: {item['source_file']} / {item['chunk_id']}\n"
            f"İçerik: {item['text']}"
        )

    return "\n\n".join(context_parts)


def get_rag_sources_json(rag_results):
    sources = []

    for item in rag_results:
        sources.append(
            {
                "source_file": item["source_file"],
                "source_title": item["source_title"],
                "chunk_id": item["chunk_id"],
                "score": item["score"]
            }
        )

    return json.dumps(sources, ensure_ascii=False)


def build_direct_rag_answer(question, rag_results):
    if not rag_results:
        return (
            "Bu konuda bilgi tabanımda yeterli bilgi bulamadım. "
            "Şu an yalnızca günlük planlama, odaklanma, mola yönetimi, plan defteri, günlük rutin, "
            "basit yemek fikirleri ve hava durumuna göre hazırlık konularındaki bilgi tabanıma göre cevap verebilirim."
        )

    best_result = rag_results[0]
    source_file = best_result["source_file"]
    text = best_result["text"]

    answer = (
        f"Bilgi tabanıma göre:\n"
        f"{text}\n\n"
        f"Kaynak: {source_file}"
    )

    return answer