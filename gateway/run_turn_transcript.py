"""Build durable inbound-turn rows, including delegated provenance."""

from typing import Any

from gateway.response_filters import delegated_continuation_metadata


def user_transcript_entry(event: Any, prepared: Any, ts: float) -> dict[str, Any]:
    """Transcript row for the inbound user turn (clean text + event time when captured)."""
    # Transient failure (429/timeout/5xx): persist the user message so the next message can load a
    # transcript that reflects what was said. The caller pairs it with a stable assistant safety
    # boundary rather than the provider error text. Hidden-reasoning-only incomplete turns follow the
    # same persistence rule so peer-agent channels don't ingest provider details. (#7100, #51628)
    _user_entry = {
        "role": "user",
        "content": (
            prepared.persist_user_message if prepared.persist_user_message is not None
            else prepared.message_text
        ),
        "timestamp": prepared.persist_user_timestamp if prepared.persist_user_timestamp is not None else ts,
    }
    if prepared.persist_user_display_kind:
        _user_entry["display_kind"] = prepared.persist_user_display_kind
    _display_metadata = {
        **({"gateway_input_owner": prepared.persistence_owner} if prepared.persistence_owner else {}),
        **delegated_continuation_metadata(event),
    }
    if _display_metadata:
        _user_entry["display_metadata"] = _display_metadata
    if getattr(event, "message_id", None):
        _user_entry["message_id"] = str(event.message_id)
    return _user_entry
