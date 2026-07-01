import sqlite3
from datetime import datetime, timedelta


DB_NAME = "bot_messages.db"
SESSION_TIMEOUT_MINUTES = 60


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE,
            user_id INTEGER,
            username TEXT,
            summary TEXT DEFAULT '',
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id INTEGER,
            username TEXT,
            first_name TEXT,
            question TEXT,
            answer TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def create_session_id(user_id):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{user_id}_{timestamp}"


def get_or_create_session(user_id, username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT session_id, last_activity_at
        FROM sessions
        WHERE user_id = ? AND status = 'active'
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,)
    )

    row = cursor.fetchone()

    if row:
        session_id = row[0]
        last_activity_at_text = row[1]

        try:
            last_activity_at = datetime.fromisoformat(last_activity_at_text)
        except Exception:
            last_activity_at = datetime.now()

        timeout_limit = datetime.now() - timedelta(minutes=SESSION_TIMEOUT_MINUTES)

        if last_activity_at >= timeout_limit:
            cursor.execute(
                """
                UPDATE sessions
                SET last_activity_at = CURRENT_TIMESTAMP,
                    username = ?
                WHERE session_id = ?
                """,
                (username, session_id)
            )

            conn.commit()
            conn.close()

            return session_id

        cursor.execute(
            """
            UPDATE sessions
            SET status = 'closed'
            WHERE session_id = ?
            """,
            (session_id,)
        )

    new_session_id = create_session_id(user_id)

    cursor.execute(
        """
        INSERT INTO sessions (
            session_id,
            user_id,
            username,
            summary,
            status
        )
        VALUES (?, ?, ?, '', 'active')
        """,
        (new_session_id, user_id, username)
    )

    conn.commit()
    conn.close()

    return new_session_id


def update_session_activity(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE sessions
        SET last_activity_at = CURRENT_TIMESTAMP
        WHERE session_id = ?
        """,
        (session_id,)
    )

    conn.commit()
    conn.close()


def save_interaction(session_id, user_id, username, first_name, question, answer):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO interactions (
            session_id,
            user_id,
            username,
            first_name,
            question,
            answer
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            first_name,
            question,
            answer
        )
    )

    conn.commit()
    conn.close()

    update_session_activity(session_id)


def save_service_log(session_id, user_id, username, service_name, request_data, response_data):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO service_logs (
            session_id,
            user_id,
            username,
            service_name,
            request_data,
            response_data
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            service_name,
            request_data,
            response_data
        )
    )

    conn.commit()
    conn.close()


def save_rag_log(session_id, user_id, username, question, source_files, matched_text):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO rag_logs (
            session_id,
            user_id,
            username,
            question,
            source_files,
            matched_text
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            user_id,
            username,
            question,
            source_files,
            matched_text
        )
    )

    conn.commit()
    conn.close()


def get_session_summary(session_id):
    conn = sqlite3.connect(DB_NAME)
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

    return row[0] or ""


def update_session_summary(session_id, summary):
    conn = sqlite3.connect(DB_NAME)
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
    conn = sqlite3.connect(DB_NAME)
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

    rows.reverse()

    interactions = []

    for row in rows:
        interactions.append(
            {
                "question": row[0],
                "answer": row[1],
                "created_at": row[2]
            }
        )

    return interactions


def count_session_interactions(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT COUNT(*)
        FROM interactions
        WHERE session_id = ?
        """,
        (session_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if not row:
        return 0

    return row[0]


def get_full_session_transcript(session_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT question, answer
        FROM interactions
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    transcript_parts = []

    for question, answer in rows:
        transcript_parts.append(f"Kullanıcı: {question}")
        transcript_parts.append(f"Bot: {answer}")

    return "\n".join(transcript_parts)