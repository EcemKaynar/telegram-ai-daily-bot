import os
import sqlite3
import uuid
from datetime import datetime, timedelta


DB_NAME = "bot_messages.db"
SESSION_TIMEOUT_MINUTES = int(os.getenv("SESSION_TIMEOUT_MINUTES", "5"))


def get_now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_datetime(value):
    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except Exception:
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_session_id(user_id):
    return f"{user_id}_{uuid.uuid4().hex[:12]}"


def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()

    for column in columns:
        if column["name"] == column_name:
            return True

    return False


def add_column_if_missing(cursor, table_name, column_name, column_definition):
    if not column_exists(cursor, table_name, column_name):
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT,
            summary TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            username TEXT,
            first_name TEXT,
            question TEXT,
            answer TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS service_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id INTEGER,
            username TEXT,
            service_name TEXT,
            request_data TEXT,
            response_data TEXT,
            duration_ms REAL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id INTEGER,
            username TEXT,
            question TEXT,
            source_files TEXT,
            matched_text TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    add_column_if_missing(cursor, "service_logs", "duration_ms", "REAL")

    conn.commit()
    conn.close()


def get_or_create_session(user_id, username):
    conn = get_connection()
    cursor = conn.cursor()

    now_text = get_now_text()
    now_dt = datetime.now()
    timeout_limit = now_dt - timedelta(minutes=SESSION_TIMEOUT_MINUTES)

    cursor.execute(
        """
        SELECT *
        FROM sessions
        WHERE user_id = ?
          AND status = 'active'
        ORDER BY last_activity_at DESC
        LIMIT 1
        """,
        (user_id,)
    )

    row = cursor.fetchone()

    if row:
        last_activity_dt = parse_datetime(row["last_activity_at"])

        if last_activity_dt and last_activity_dt >= timeout_limit:
            session_id = row["session_id"]

            cursor.execute(
                """
                UPDATE sessions
                SET last_activity_at = ?, username = ?
                WHERE session_id = ?
                """,
                (now_text, username, session_id)
            )

            conn.commit()
            conn.close()

            return session_id

        cursor.execute(
            """
            UPDATE sessions
            SET status = 'expired'
            WHERE session_id = ?
            """,
            (row["session_id"],)
        )

    session_id = create_session_id(user_id)

    cursor.execute(
        """
        INSERT INTO sessions (
            session_id,
            user_id,
            username,
            summary,
            status,
            created_at,
            last_activity_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            "",
            "active",
            now_text,
            now_text
        )
    )

    conn.commit()
    conn.close()

    return session_id


def update_session_activity(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET last_activity_at = ?
        WHERE session_id = ?
        """,
        (get_now_text(), session_id)
    )

    conn.commit()
    conn.close()


def save_interaction(session_id, user_id, username, first_name, question, answer):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO interactions (
            session_id,
            user_id,
            username,
            first_name,
            question,
            answer,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            first_name,
            question,
            answer,
            get_now_text()
        )
    )

    cursor.execute(
        """
        UPDATE sessions
        SET last_activity_at = ?
        WHERE session_id = ?
        """,
        (get_now_text(), session_id)
    )

    conn.commit()
    conn.close()


def save_service_log(
    session_id,
    user_id,
    username,
    service_name,
    request_data,
    response_data,
    duration_ms=None
):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO service_logs (
            session_id,
            user_id,
            username,
            service_name,
            request_data,
            response_data,
            duration_ms,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            service_name,
            request_data,
            response_data,
            duration_ms,
            get_now_text()
        )
    )

    conn.commit()
    conn.close()


def save_rag_log(session_id, user_id, username, question, source_files, matched_text):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO rag_logs (
            session_id,
            user_id,
            username,
            question,
            source_files,
            matched_text,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            question,
            source_files,
            matched_text,
            get_now_text()
        )
    )

    conn.commit()
    conn.close()


def get_session_summary(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT summary
        FROM sessions
        WHERE session_id = ?
        """,
        (session_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return ""

    return row["summary"] or ""


def update_session_summary(session_id, summary):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET summary = ?
        WHERE session_id = ?
        """,
        (summary, session_id)
    )

    conn.commit()
    conn.close()


def get_recent_interactions(session_id, limit=6):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT question, answer, created_at
        FROM interactions
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (session_id, limit)
    )

    rows = cursor.fetchall()
    conn.close()

    results = [dict(row) for row in rows]
    results.reverse()

    return results


def count_session_interactions(session_id):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM interactions
        WHERE session_id = ?
        """,
        (session_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return 0

    return row["total"]


def get_full_session_transcript(session_id):
    interactions = get_recent_interactions(session_id, limit=50)

    lines = []

    for item in interactions:
        lines.append(f"User: {item.get('question', '')}")
        lines.append(f"Assistant: {item.get('answer', '')}")

    return "\n".join(lines)