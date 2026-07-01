import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


from webhook_bot import generate_ai_answer, looks_like_weather_question


TEST_CASES_FILE = os.path.join(PROJECT_ROOT, "evaluation", "test_cases.json")
RESULTS_FILE = os.path.join(PROJECT_ROOT, "evaluation", "evaluation_results.csv")

TEST_TIMEOUT_SECONDS = 90


def load_test_cases():
    with open(TEST_CASES_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_text(text):
    if text is None:
        return ""

    return str(text).lower().strip()


def contains_any(text, words):
    text = normalize_text(text)

    if not words:
        return True

    for word in words:
        if normalize_text(word) in text:
            return True

    return False


def contains_none(text, words):
    text = normalize_text(text)

    if not words:
        return True

    for word in words:
        if normalize_text(word) in text:
            return False

    return True


def run_bot_answer(user_text, test_id):
    session_id = f"eval_{test_id}_{int(time.time())}"

    return generate_ai_answer(
        user_text=user_text,
        session_id=session_id,
        history_context="",
        user_id=0,
        username="evaluation_user"
    )


def run_with_timeout(func, timeout_seconds):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)

        try:
            return future.result(timeout=timeout_seconds), None
        except TimeoutError:
            return None, "timeout"
        except Exception as error:
            return None, str(error)


def score_test_case(test_case, answer):
    score = 0
    max_score = 4
    notes = []

    expected_type = test_case["expected_type"]
    answer_text = normalize_text(answer)

    if expected_type == "rag":
        if "kaynak:" in answer_text:
            score += 1
            notes.append("Kaynak gösterildi")
        else:
            notes.append("Kaynak gösterilmedi")

        expected_source = test_case.get("expected_source")

        if expected_source and normalize_text(expected_source) in answer_text:
            score += 1
            notes.append("Beklenen kaynak bulundu")
        else:
            notes.append("Beklenen kaynak bulunamadı")

    elif expected_type == "unknown":
        if "bilgi tabanımda" in answer_text or "yeterli bilgi bulamadım" in answer_text:
            score += 2
            notes.append("Bilgi yok cevabı doğru")
        else:
            notes.append("Bilgi yok cevabı eksik")

    elif expected_type == "weather":
        if looks_like_weather_question(test_case["input"]):
            score += 1
            notes.append("Tool gerektiren soru algılandı")
        else:
            notes.append("Tool gerektiren soru algılanmadı")

        if "bilgi tabanımda yeterli bilgi bulamadım" not in answer_text:
            score += 1
            notes.append("Weather sorusu RAG yok cevabına düşmedi")
        else:
            notes.append("Weather sorusu yanlışlıkla RAG yok cevabına düştü")

    if contains_any(answer, test_case.get("must_include_any", [])):
        score += 1
        notes.append("Beklenen anahtar ifadelerden en az biri var")
    else:
        notes.append("Beklenen anahtar ifadeler yok")

    if contains_none(answer, test_case.get("must_not_include_any", [])):
        score += 1
        notes.append("Yasaklı ifadeler yok")
    else:
        notes.append("Yasaklı ifade bulundu")

    success_rate = round((score / max_score) * 100, 2)

    return score, max_score, success_rate, " | ".join(notes)


def save_results(results):
    fieldnames = [
        "test_id",
        "category",
        "input",
        "expected_type",
        "score",
        "max_score",
        "success_rate",
        "status",
        "answer",
        "notes",
        "created_at"
    ]

    with open(RESULTS_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results):
    total_tests = len(results)
    passed_tests = len([result for result in results if result["status"] == "PASS"])
    failed_tests = total_tests - passed_tests

    average_score = 0

    if total_tests > 0:
        average_score = sum(float(result["success_rate"]) for result in results) / total_tests

    print("\n===== DEĞERLENDİRME ÖZETİ =====")
    print(f"Toplam test: {total_tests}")
    print(f"Başarılı test: {passed_tests}")
    print(f"Başarısız test: {failed_tests}")
    print(f"Ortalama başarı: {round(average_score, 2)}%")
    print(f"Sonuç dosyası: {RESULTS_FILE}")
    print("================================\n")


def main():
    test_cases = load_test_cases()
    results = []

    print("Değerlendirme başlatıldı...\n")

    for test_case in test_cases:
        test_id = test_case["id"]
        user_input = test_case["input"]

        print(f"Test çalışıyor: {test_id}")
        print(f"Soru: {user_input}")

        started_at = time.time()

        answer, error = run_with_timeout(
            func=lambda: run_bot_answer(user_input, test_id),
            timeout_seconds=TEST_TIMEOUT_SECONDS
        )

        elapsed = round(time.time() - started_at, 2)

        if error:
            answer = f"ERROR: {error}"
            score = 0
            max_score = 4
            success_rate = 0
            notes = f"Test çalışırken hata oluştu: {error}"
        else:
            score, max_score, success_rate, notes = score_test_case(
                test_case=test_case,
                answer=answer
            )

        status = "PASS" if success_rate >= 75 else "FAIL"

        result = {
            "test_id": test_id,
            "category": test_case["category"],
            "input": user_input,
            "expected_type": test_case["expected_type"],
            "score": score,
            "max_score": max_score,
            "success_rate": success_rate,
            "status": status,
            "answer": answer,
            "notes": notes,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        results.append(result)

        print(f"Cevap: {answer}")
        print(f"Skor: {score}/{max_score} - {success_rate}% - {status}")
        print(f"Süre: {elapsed} saniye")
        print("-" * 50)

    save_results(results)
    print_summary(results)


if __name__ == "__main__":
    main()