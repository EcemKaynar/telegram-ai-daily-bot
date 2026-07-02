from database import (
    get_or_create_session,
    get_session_summary,
    get_recent_interactions,
    count_session_interactions,
    get_full_session_transcript,
    update_session_summary
)

from openrouter_client import summarize_session


SUMMARY_UPDATE_INTERVAL = 1000


def build_history_context(session_id, recent_limit=6):
    summary = get_session_summary(session_id)
    recent_interactions = get_recent_interactions(session_id, limit=recent_limit)

    parts = []

    if summary:
        parts.append("Session summary:")
        parts.append(summary)

    if recent_interactions:
        parts.append("Recent conversation:")

        for item in recent_interactions:
            question = item.get("question", "")
            answer = item.get("answer", "")

            parts.append(f"User: {question}")
            parts.append(f"Assistant: {answer}")

    return "\n".join(parts).strip()


def get_session_and_history(user_id, username):
    session_id = get_or_create_session(
        user_id=user_id,
        username=username
    )

    history_context = build_history_context(
        session_id=session_id,
        recent_limit=6
    )

    return session_id, history_context


def refresh_session_summary_if_needed(session_id):
    try:
        interaction_count = count_session_interactions(session_id)

        if interaction_count == 0:
            return

        if interaction_count % SUMMARY_UPDATE_INTERVAL != 0:
            return

        transcript = get_full_session_transcript(session_id)

        if not transcript.strip():
            return

        summary = summarize_session(transcript)

        if summary:
            update_session_summary(session_id, summary)

    except Exception as error:
        print(f"Session summary güncellenemedi: {error}")