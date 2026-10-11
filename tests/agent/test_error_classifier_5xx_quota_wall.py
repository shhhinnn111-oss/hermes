"""A 503/529 that reports spent quota falls back instead of backing off on the same model.

Credential-pooling gateways (e.g. CLIProxyAPI) answer 503 once every upstream credential for the
model is spent. Same-model backoff cannot succeed, and the key to the gateway is healthy (other
models still work), so the verdict falls back without benching it. Capacity 503/529s keep backing off.
"""
import time
from datetime import UTC, datetime
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from agent.error_classifier import FailoverReason, classify_api_error

_POOL_WALL = (
    "auth_unavailable: no auth available (providers=codex, model=gpt-6-astra; "
    "last upstream error: usage_limit_reached: The usage limit has been reached)"
)


class _APIError(Exception):
    def __init__(self, message, status_code, body=None, headers=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body or {}
        self.response = SimpleNamespace(headers=headers or {})


def _classify(message, status, **kwargs):
    return classify_api_error(_APIError(message, status, **kwargs), provider="custom", model="gpt-6-astra")


def _assert_falls_back_without_benching(result):
    assert result.reason == FailoverReason.upstream_rate_limit
    assert result.should_fallback is True
    assert result.should_rotate_credential is False
    assert result.retryable is True


@pytest.mark.parametrize("status", [503, 529])
def test_pool_quota_wall_falls_back_without_benching_key(status):
    result = _classify(_POOL_WALL, status, body={"error": {"message": _POOL_WALL, "type": "server_error"}})
    _assert_falls_back_without_benching(result)
    assert result.error_context == {}


@pytest.mark.parametrize("status", [503, 529])
@pytest.mark.parametrize("suffix", [". Please try again in 2 hours", ". Please retry after a while"])
def test_quota_wall_with_retry_wording_still_falls_back(status, suffix):
    _assert_falls_back_without_benching(_classify(_POOL_WALL + suffix, status))


@pytest.mark.parametrize("status", [503, 529])
def test_quota_wall_carries_reset_window_from_message(status):
    before = time.time()
    result = _classify(_POOL_WALL + "; resets in 1 hour", status)
    _assert_falls_back_without_benching(result)
    assert before + 3590 <= result.error_context["reset_at"] <= time.time() + 3610


def test_quota_wall_carries_retry_after_header_window():
    before = time.time()
    result = _classify(_POOL_WALL, 503, headers={"retry-after": "120"})
    _assert_falls_back_without_benching(result)
    assert before + 110 <= result.error_context["reset_at"] <= time.time() + 130


def test_structured_usage_limit_code_falls_back():
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"}}
    _assert_falls_back_without_benching(_classify("Service Unavailable", 503, body=body))


@pytest.mark.parametrize("status, message", [
    (503, "Service temporarily overloaded, please retry"),
    (529, "Service temporarily overloaded, please retry"),
    # Capacity, not exhaustion: overload wording wins, and bare "quota"/"limit exceeded" is not enough.
    (503, "Service temporarily overloaded: concurrency limit exceeded"),
    (503, "concurrency limit exceeded"),
    (529, "Overloaded: quota service temporarily unavailable"),
    (529, "overloaded_error: Overloaded (usage limit scheduler busy)"),
])
def test_capacity_errors_still_back_off(status, message):
    result = classify_api_error(_APIError(message, status))
    assert result.reason == FailoverReason.overloaded
    assert result.retryable is True
    assert result.should_fallback is False


_FUTURE = time.time() + 86400  # stays in the future for the whole test session


@pytest.mark.parametrize("extracted, expected", [({}, "classifier"), ({"reset_at": _FUTURE}, _FUTURE)])
def test_eager_fallback_gets_the_reset_window(monkeypatch, extracted, expected):
    """The turn's separately extracted context may miss a reset the classifier found (nested
    bodies, long messages); the fallback cooldown must still get it."""
    from agent import conversation_loop
    from agent.turn_recovery import route_classified_error

    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *args: "prompt")
    seen = {}
    agent = SimpleNamespace(
        _fallback_index=0, _fallback_chain=[{"provider": "fallback"}], _credential_pool=None,
        _buffer_diagnostic_status=lambda *_: None,
        _try_activate_fallback=lambda **kwargs: seen.update(kwargs) or True,
    )
    error = _APIError(_POOL_WALL + "; resets in 1 hour", 503)
    classified = classify_api_error(error, provider="custom", model="gpt-6-astra")
    verdict = route_classified_error(
        agent, error, classified, SimpleNamespace(), error_msg=str(error), error_context=extracted,
        recovered_with_pool=False, base_url="http://127.0.0.1:8317/v1", model="gpt-6-astra",
        messages=[], api_messages=[], system_message="", active_system_prompt="",
        conversation_history=[], retry_count=0, max_retries=3, compression_attempts=0,
        max_compression_attempts=3, api_call_count=1, effective_task_id="task",
    )
    assert verdict.action == "break"
    assert seen["reason"] == FailoverReason.upstream_rate_limit
    assert seen["reset_at"] == (classified.error_context["reset_at"] if expected == "classifier" else expected)


def _absolute_reset_forms(at):
    return [
        str(int(at)), int(at * 1000), str(int(at * 1000)),
        datetime.fromtimestamp(at, UTC).isoformat(),
        format_datetime(datetime.fromtimestamp(at, UTC), usegmt=True),
    ]


@pytest.mark.parametrize("form", range(5))
def test_absolute_reset_fields_are_timestamps_not_durations(form):
    """``resets_at``/``reset_at`` are absolute (epoch s/ms or ISO, numeric or string, or an HTTP-date):
    a string epoch read as a duration would bench the primary for decades."""
    at = time.time() + 3600
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"},
            "resets_at": _absolute_reset_forms(at)[form]}
    result = _classify("Service Unavailable", 503, body=body)
    _assert_falls_back_without_benching(result)
    assert at - 5 <= result.error_context["reset_at"] <= at + 5


def test_eager_fallback_gets_a_top_level_string_epoch_reset(monkeypatch):
    """Real turn flow: the extractor reads only ``body["error"]``, so this reset comes from the
    classifier alone; the cooldown must get the instant, not ~1.7e9 s."""
    from agent import conversation_loop
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.turn_recovery import route_classified_error

    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *args: "prompt")
    seen = {}
    agent = SimpleNamespace(
        _fallback_index=0, _fallback_chain=[{"provider": "fallback"}], _credential_pool=None,
        _buffer_diagnostic_status=lambda *_: None,
        _try_activate_fallback=lambda **kwargs: seen.update(kwargs) or True,
    )
    at = time.time() + 3600
    error = _APIError("Service Unavailable", 503, body={
        "error": {"code": "usage_limit_reached", "message": "Service Unavailable"}, "resets_at": str(int(at)),
    })
    route_classified_error(
        agent, error, classify_api_error(error, provider="custom", model="m"), SimpleNamespace(),
        error_msg=str(error), error_context=extract_api_error_context(error), recovered_with_pool=False,
        base_url="http://127.0.0.1:8317/v1", model="m", messages=[], api_messages=[], system_message="",
        active_system_prompt="", conversation_history=[], retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="task",
    )
    assert at - 5 <= seen["reset_at"] <= at + 5


def test_past_absolute_reset_defers_to_retry_after():
    body = {"error": {"code": "usage_limit_reached", "message": "Service Unavailable"},
            "resets_at": str(int(time.time() - 600))}
    before = time.time()
    result = _classify("Service Unavailable", 503, body=body, headers={"retry-after": "120"})
    assert before + 110 <= result.error_context["reset_at"] <= time.time() + 130



@pytest.mark.parametrize("make_reset, headers, window", [
    (lambda now: int(now - 600), {"retry-after": "120"}, 120),
    (lambda now: str(int(now - 600)), {"retry-after": "120"}, 120),
    (lambda now: format_datetime(datetime.fromtimestamp(now + 3600, UTC), usegmt=True), {}, 3600),
], ids=["past-epoch", "past-string-epoch", "future-http-date"])
def test_eager_fallback_ignores_an_extracted_reset_that_is_not_a_future_instant(
    monkeypatch, make_reset, headers, window
):
    """The extractor reads ``body["error"]["resets_at"]`` as-is; an expired or unparseable value must
    not override the classifier's reset and leave the cooldown at its 60 s default."""
    from agent import conversation_loop
    from agent.agent_runtime_helpers import extract_api_error_context
    from agent.fallback_cooldown import _provider_reset_delay
    from agent.turn_recovery import route_classified_error

    monkeypatch.setattr(conversation_loop, "_arm_fallback_restart", lambda *args: "prompt")
    seen = {}
    agent = SimpleNamespace(
        _fallback_index=0, _fallback_chain=[{"provider": "fallback"}], _credential_pool=None,
        _buffer_diagnostic_status=lambda *_: None,
        _try_activate_fallback=lambda **kwargs: seen.update(kwargs) or True,
    )
    error = _APIError("Service Unavailable", 503, headers=headers, body={
        "error": {"code": "usage_limit_reached", "message": "Service Unavailable", "resets_at": make_reset(time.time())},
    })
    route_classified_error(
        agent, error, classify_api_error(error, provider="custom", model="m"), SimpleNamespace(),
        error_msg=str(error), error_context=extract_api_error_context(error), recovered_with_pool=False,
        base_url="http://127.0.0.1:8317/v1", model="m", messages=[], api_messages=[], system_message="",
        active_system_prompt="", conversation_history=[], retry_count=0, max_retries=3,
        compression_attempts=0, max_compression_attempts=3, api_call_count=1, effective_task_id="task",
    )
    assert window - 15 <= _provider_reset_delay(seen["reset_at"]) <= window + 5
