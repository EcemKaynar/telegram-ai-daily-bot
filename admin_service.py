import csv
import os
import sqlite3

from database import DB_NAME


EVALUATION_RESULTS_FILE = os.path.join("evaluation", "evaluation_results.csv")


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def rows_to_dicts(rows):
    return [dict(row) for row in rows]


def safe_count(table_name):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) AS total FROM {table_name}")
        row = cursor.fetchone()

        conn.close()

        return row["total"] if row else 0

    except Exception:
        return 0


def get_admin_overview():
    sessions_count = safe_count("sessions")
    interactions_count = safe_count("interactions")
    service_logs_count = safe_count("service_logs")
    rag_logs_count = safe_count("rag_logs")

    evaluation_summary = get_evaluation_summary()

    return {
        "sessions_count": sessions_count,
        "interactions_count": interactions_count,
        "service_logs_count": service_logs_count,
        "rag_logs_count": rag_logs_count,
        "evaluation_average_success": evaluation_summary["average_success"],
        "evaluation_total_tests": evaluation_summary["total_tests"],
        "evaluation_passed_tests": evaluation_summary["passed_tests"],
        "evaluation_failed_tests": evaluation_summary["failed_tests"]
    }


def get_sessions():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                s.session_id,
                s.user_id,
                s.username,
                s.summary,
                s.status,
                s.created_at,
                s.last_activity_at,
                COUNT(i.id) AS message_count
            FROM sessions s
            LEFT JOIN interactions i ON s.session_id = i.session_id
            GROUP BY s.session_id
            ORDER BY s.last_activity_at DESC
            """
        )

        rows = cursor.fetchall()
        conn.close()

        return rows_to_dicts(rows)

    except Exception as error:
        print(f"Session listesi alınamadı: {error}")
        return []


def get_session_detail(session_id):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                session_id,
                user_id,
                username,
                summary,
                status,
                created_at,
                last_activity_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,)
        )

        session_row = cursor.fetchone()

        cursor.execute(
            """
            SELECT
                id,
                session_id,
                user_id,
                username,
                first_name,
                question,
                answer,
                created_at
            FROM interactions
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,)
        )

        interaction_rows = cursor.fetchall()

        conn.close()

        return {
            "session": dict(session_row) if session_row else None,
            "interactions": rows_to_dicts(interaction_rows)
        }

    except Exception as error:
        print(f"Session detayı alınamadı: {error}")
        return {
            "session": None,
            "interactions": []
        }


def get_service_logs(session_id=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_id,
                    username,
                    service_name,
                    request_data,
                    response_data,
                    created_at
                FROM service_logs
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (session_id,)
            )
        else:
            cursor.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_id,
                    username,
                    service_name,
                    request_data,
                    response_data,
                    created_at
                FROM service_logs
                ORDER BY id DESC
                LIMIT 100
                """
            )

        rows = cursor.fetchall()
        conn.close()

        return rows_to_dicts(rows)

    except Exception as error:
        print(f"Service logs alınamadı: {error}")
        return []


def get_rag_logs(session_id=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_id,
                    username,
                    question,
                    source_files,
                    matched_text,
                    created_at
                FROM rag_logs
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT 100
                """,
                (session_id,)
            )
        else:
            cursor.execute(
                """
                SELECT
                    id,
                    session_id,
                    user_id,
                    username,
                    question,
                    source_files,
                    matched_text,
                    created_at
                FROM rag_logs
                ORDER BY id DESC
                LIMIT 100
                """
            )

        rows = cursor.fetchall()
        conn.close()

        return rows_to_dicts(rows)

    except Exception as error:
        print(f"RAG logs alınamadı: {error}")
        return []


def get_evaluation_results():
    if not os.path.exists(EVALUATION_RESULTS_FILE):
        return []

    try:
        with open(EVALUATION_RESULTS_FILE, "r", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            return list(reader)

    except Exception as error:
        print(f"Evaluation results okunamadı: {error}")
        return []


def get_evaluation_summary():
    results = get_evaluation_results()

    total_tests = len(results)

    if total_tests == 0:
        return {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "average_success": 0
        }

    passed_tests = len([
        result
        for result in results
        if result.get("status") == "PASS"
    ])

    failed_tests = total_tests - passed_tests

    success_values = []

    for result in results:
        try:
            success_values.append(float(result.get("success_rate", 0)))
        except Exception:
            success_values.append(0)

    average_success = round(sum(success_values) / total_tests, 2)

    return {
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "average_success": average_success
    }


def get_admin_data():
    return {
        "overview": get_admin_overview(),
        "sessions": get_sessions(),
        "service_logs": get_service_logs(),
        "rag_logs": get_rag_logs(),
        "evaluation_results": get_evaluation_results(),
        "evaluation_summary": get_evaluation_summary()
    }