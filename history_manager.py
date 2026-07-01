from database import (
    get_or_create_session,
    get_session_summary,
    get_recent_interactions,
    count_session_interactions,
    get_full_session_transcript,
    update_session_summary
)

from openrouter_client import summarize_session


SUMMARY_UPDATE_INTERVAL = 9999


def build_history_context(session_id):
    summary = get_session_summary(session_id)
    recent_interactions = get_recent_interactions(session_id, limit=8)

    context_parts = []

    if summary:
        context_parts.append("Oturum özeti:")
        context_parts.append(summary)

    if recent_interactions:
        context_parts.append("Son konuşmalar:")

        for question, answer, created_at in recent_interactions:
            context_parts.append(f"Kullanıcı: {question}")
            context_parts.append(f"Bot: {answer}")

    if not context_parts:
        return ""

    return "\n".join(context_parts)


def get_session_and_history(user_id, username):
    session_id = get_or_create_session(user_id, username)
    history_context = build_history_context(session_id)

    return session_id, history_context


def should_update_summary(session_id):
    count = count_session_interactions(session_id)

    if count == 0:
        return False

    return count % SUMMARY_UPDATE_INTERVAL == 0


def refresh_session_summary_if_needed(session_id):
    if not should_update_summary(session_id):
        return

    previous_summary = get_session_summary(session_id)
    full_transcript = get_full_session_transcript(session_id)

    transcript_text_parts = []

    for question, answer, created_at in full_transcript:
        transcript_text_parts.append(f"Kullanıcı: {question}")
        transcript_text_parts.append(f"Bot: {answer}")

    transcript_text = "\n".join(transcript_text_parts)

    new_summary = summarize_session(
        previous_summary=previous_summary,
        transcript_text=transcript_text
    )

    update_session_summary(session_id, new_summary)