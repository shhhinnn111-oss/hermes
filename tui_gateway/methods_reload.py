"""``reload.mcp`` and ``skills.reload`` bodies: a reload acts for the CALLER's profile only (#58414).

The handlers in ``methods_tools`` bind the caller's profile scope (``@_profile_scoped``: the live
session's ``profile_home``, or ``profile`` for sessionless calls) and delegate here. Each profile
owns its MCP connections and credentials, so a reload from profile A cannot change what profile B
can reach; reloading B too only cost B its prompt cache, its in-flight MCP calls and, when a server
failed to reconnect, its tools. Bodies are rebound onto ``server.py``'s globals (``method_ctx``).
"""

import contextlib
import hashlib
import json
import threading

from .method_ctx import bind_module

# Serialize reload.mcp (runs on the pool): overlapping shutdown+discover pairs would leave the registry half-built.
_mcp_reload_lock = threading.Lock()
# Per profile home key. Bumped per SUCCESSFUL reload; a follower skips only if it advanced while it waited
# (a leader that threw leaves it unchanged → the follower reloads itself).
_mcp_reload_gen: dict[str, int] = {}
# The mcp_rev the last successful reload of that home actually LOADED (re-hashed after discovery); a
# follower coalesces only when its requested rev matches, otherwise the config changed under the leader.
_mcp_reload_loaded_rev: dict[str, str] = {}
# Bounded convergence for a config edit racing a slow reload: the leader re-hashes until the hash is stable.
_MCP_RELOAD_MAX_PASSES = 3
# One reload reconnects every server of the profile; the host's 30 s control default undercuts the client.
_MCP_RELOAD_HOST_TIMEOUT_S = 120.0


def _compute_mcp_rev() -> str:
    """Hash of mcp_servers (definitions — omitting it meant an edited server never bumped the rev) + mcp +
    tools. ``config.get mtime`` ships it so cosmetic writes don't reload; ``reload.mcp`` coalesces on it. "" = unknown."""
    with contextlib.suppress(Exception):
        cfg = _load_cfg()
        rev_src = json.dumps({k: cfg.get(k) for k in ("mcp", "mcp_servers", "tools")}, sort_keys=True, default=str)
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    return ""


def _mcp_reload_confirm_required() -> bool:
    """``approvals.mcp_reload_confirm`` from disk config; True (safe) on any failure."""
    try:
        cfg = _tools_mod("hermes_cli.config").load_config()
        approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
        return bool(approvals.get("mcp_reload_confirm", True)) if isinstance(approvals, dict) else True
    except Exception:
        return True


def _queue_turn_note(agent, note: str) -> None:
    """Queue ``note`` for the agent's next turn on the one-shot turn-note channel (``agent/turn_context.py``)."""
    if agent is None or not note:
        return
    prior = getattr(agent, "_gateway_turn_context_notes", "") or ""
    agent._gateway_turn_context_notes = f"{prior}\n\n{note}" if prior else note


def _sessions_of_homes(homes: set) -> list:
    """``(sid, session)`` of the given profile home keys (a session with no ``profile_home`` is the launch home's)."""
    from hermes_constants import hermes_home_key
    with _sessions_lock:
        return [(sid, sess) for sid, sess in _sessions.items()
                if hermes_home_key(sess.get("profile_home") or get_process_hermes_home()) in homes]


def _adopter_homes() -> set:
    """Profiles that adopted one of the bound profile's live connections (``mcp_tool_scope``): the
    teardown closes those connections under them too."""
    mcp_core = _tools_mod("tools.mcp_tool")
    scope = mcp_core._mcp_registry_scope()
    if scope is None:
        return set()
    with mcp_core._lock:
        return {adopter for key, owner in mcp_core._server_scope_keys.items() if owner == scope
                for adopter in mcp_core._server_tool_scopes.get(key, ()) if adopter != scope}


def _shutdown_profile_mcp_servers() -> None:
    """Tear down the bound profile's MCP connections. A single-profile process keeps the process-wide
    teardown. Under a multiplexer each profile owns ``(scope, name)`` keys and the launch profile also owns
    the bare keys it connected before the process began to multiplex, pruned by name: a bare call is the
    wildcard that tears down every profile."""
    from hermes_constants import hermes_home_key
    mcp_core, lifecycle = _tools_mod("tools.mcp_tool"), _tools_mod("tools.mcp_tool_lifecycle")
    scope = mcp_core._mcp_registry_scope()
    if scope is None:
        lifecycle.shutdown_mcp_servers()
        return
    lifecycle.shutdown_mcp_servers(scope=scope)
    if hermes_home_key() != hermes_home_key(get_process_hermes_home()):
        return
    with mcp_core._lock:
        bare = {key for key in mcp_core._servers if isinstance(key, str)}
    if bare:
        lifecycle.shutdown_mcp_servers(scope=None, names=bare)


def _failed_mcp_servers() -> list:
    """Names of the bound profile's servers whose connect failed. Names only: the error text can quote a
    URL or header the server echoed back."""
    mcp_core, mcp_scope = _tools_mod("tools.mcp_tool"), _tools_mod("tools.mcp_tool_scope")
    scope = mcp_core._mcp_registry_scope()
    with mcp_core._lock:
        return sorted({mcp_scope._key_name(key) for key in mcp_core._server_connect_errors
                       if mcp_core._server_visible_in_scope(key, scope)})


def _finish_reload(rid, params: dict, payload: dict) -> dict:
    """Shared tail for every successful reload: honor ``always`` (persist the confirm opt-out into the
    bound profile's config) and return the ok payload."""
    if bool(params.get("always", False)):
        try:
            from cli import save_config_value
            save_config_value("approvals.mcp_reload_confirm", False)
        except Exception as _exc:
            logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)
    return _ok(rid, payload)


def _reload_mcp_on_host(rid, params: dict) -> dict:
    """Turn-isolated session: the child owns the MCP pool; only a confirmed nested success is success."""
    try:
        ack = _get_compute_host_supervisor().reload_mcp(
            str(params.get("session_id") or ""), request_id=f"reload-mcp-{rid}",
            rev=str(params.get("rev") or ""), timeout=_MCP_RELOAD_HOST_TIMEOUT_S)
    except Exception as exc:
        logger.warning("compute-host reload_mcp failed (session %s)", params.get("session_id"), exc_info=True)
        return _err(rid, 5019, f"compute-host reload_mcp failed: {exc}")
    if not isinstance(ack, dict):
        return _err(rid, 5019, "compute-host reload_mcp failed: malformed host acknowledgement")
    response = ack.get("response")
    result = response.get("result") if isinstance(response, dict) else None
    if ack.get("type") != "reload_mcp.ack" or not isinstance(result, dict) or result.get("status") != "reloaded":
        error = response.get("error") if isinstance(response, dict) else None
        message = (error.get("message") if isinstance(error, dict) else None) or ack.get("message")
        return _err(rid, 5019, f"compute-host reload_mcp failed: {message or 'no successful host acknowledgement'}")
    return _finish_reload(rid, params, {**result, "turn_isolation": True, "host_ack": ack})


def _reload_mcp(rid, params: dict) -> dict:
    """Body of ``reload.mcp``, run inside the caller's profile scope. A turn running on the pool being
    reloaded refuses it: the teardown would fail its in-flight MCP calls and swap its tools between two API
    calls of the same turn. That pool is the compute host's own for a turn-isolated session, else the
    profile's in-process pool, which also serves the profiles that adopted one of its connections."""
    from hermes_constants import hermes_home_key
    home = hermes_home_key()
    own = _sessions_of_homes({home})
    sid = params.get("session_id", "")
    # Only a session of the bound profile: a pane may name its profile while sending another
    # profile's active chat, whose compute host reloads THAT profile.
    session = dict(own).get(sid)
    on_host = bool(session) and _session_uses_compute_host(session)
    sharing = [(sid, session)] if on_host else [
        (other_sid, sess) for other_sid, sess in own + _sessions_of_homes(_adopter_homes())
        if not _session_uses_compute_host(sess)]
    busy = sorted(other_sid for other_sid, sess in sharing if sess.get("running"))
    if busy:
        return _err(rid, 4009, f"session busy — a turn is running in {', '.join(busy)}; "
                               "reload MCP servers when it finishes")
    # Prompt-cache invalidation gate: without confirm=true honour ``approvals.mcp_reload_confirm``
    # (Ink prints ``message`` and re-invokes with confirm=true, or flips the config).
    if not bool(params.get("confirm", False)) and _mcp_reload_confirm_required():
        message = (
            "⚠️  /reload-mcp invalidates the prompt cache (next message re-sends full input tokens). "
            "Reply `/reload-mcp now` to proceed, or `/reload-mcp always` to proceed and "
            "silence this prompt permanently.")
        return _ok(rid, {"status": "confirm_required", "message": message})
    if on_host:
        return _reload_mcp_on_host(rid, params)
    # Revision the CALLER wants loaded; empty on legacy clients / manual /reload-mcp
    # (generation-only coalescing).
    req_rev = str(params.get("rev") or "")

    def _payload(coalesced: bool) -> dict:
        """Read under _mcp_reload_lock: another profile's reload clears connect errors as it starts."""
        return {"status": "reloaded", "loaded_rev": _mcp_reload_loaded_rev.get(home, ""),
                "failed": _failed_mcp_servers(), **({"coalesced": True} if coalesced else {})}

    def _do_full_reload() -> None:
        """shutdown+discover+refresh under the lock, then mark a completed generation. Config
        can change WHILE discover connects: re-hash and repeat until stable so the marked
        generation matches what loaded. Profiles that had adopted a connection this teardown
        closed re-adopt or connect their own, even when this profile no longer configures any
        server (discovery then returns before re-registering them)."""
        mcp_agent, mcp_discovery = _tools_mod("tools.mcp_tool_agent"), _tools_mod("tools.mcp_tool_discovery")
        loaded = _compute_mcp_rev()
        for _ in range(_MCP_RELOAD_MAX_PASSES):
            _shutdown_profile_mcp_servers()
            mcp_agent.reprobe_tool_availability()
            mcp_discovery.discover_mcp_tools()
            after = _compute_mcp_rev()
            if after == loaded:
                break
            loaded = after
        _tools_mod("tools.mcp_tool_lifecycle")._reregister_orphaned_adopters()
        _refresh_live_sessions(get_hermes_home())
        _mcp_reload_loaded_rev[home] = loaded
        _mcp_reload_gen[home] = _mcp_reload_gen.get(home, 0) + 1

    # LEADER (won the non-blocking acquire) runs the full reload. FOLLOWER waits, then — still
    # holding the lock — coalesces only if a reload of ITS profile COMPLETED meanwhile (generation
    # advanced ⇒ leader didn't throw) AND it loaded the requested revision; otherwise it re-runs.
    if _mcp_reload_lock.acquire(blocking=False):
        try:
            _do_full_reload()
            payload = _payload(False)
        finally:
            _mcp_reload_lock.release()
        return _finish_reload(rid, params, payload)
    gen_before = _mcp_reload_gen.get(home, 0)
    with _mcp_reload_lock:
        coalesced = (_mcp_reload_gen.get(home, 0) > gen_before
                     and (not req_rev or req_rev == _mcp_reload_loaded_rev.get(home, "")))
        if coalesced:
            _refresh_live_sessions(get_hermes_home())
        else:
            _do_full_reload()
        payload = _payload(coalesced)
    return _finish_reload(rid, params, payload)


def _skills_reload_note(result: dict) -> str:
    """Turn note for the session that asked. Names only: a description is text from a skill file the
    user may not have read, and this note lands in the model's next user turn."""
    added = [str(item.get("name", "")) for item in result.get("added") or []]
    removed = [str(item.get("name", "")) for item in result.get("removed") or []]
    if not added and not removed:
        return ""
    sections = ["[USER INITIATED SKILLS RELOAD:"]
    for header, names in (("Added Skills:", added), ("Removed Skills:", removed)):
        if names:
            sections += ["", header, *(f"    - {name}" for name in names)]
    return "\n".join([*sections, "", "Use skills_list to see the updated catalog.]"])


def _queue_skills_reload_note(session, result: dict) -> None:
    """The requesting session only, as the gateway and CLI do, and only when it belongs to the reloaded
    profile (a call may name a profile and send another profile's chat); the system prompt stays
    byte-stable."""
    from hermes_constants import hermes_home_key
    if not isinstance(session, dict):
        return
    if hermes_home_key(session.get("profile_home") or get_process_hermes_home()) != hermes_home_key():
        return
    _queue_turn_note(session.get("agent"), _skills_reload_note(result))


_RELOAD_MCP_ONCE = frozenset({"now", "approve", "once", "yes"})


def _format_live_reload_mcp_output(sid: str, session, arg: str) -> str:
    """``slash.exec /reload-mcp [now|always]`` runs the ``reload.mcp`` RPC, not the slash worker: the
    worker's MCP pool is its own, so a worker reload never reached the live session. Same argument
    words as the Ink TUI (``ui-tui/src/app/slash/commands/ops.ts``); the consent gate still applies."""
    choice = (arg or "").strip().lower()
    params = {"session_id": sid, "confirm": choice in _RELOAD_MCP_ONCE or choice == "always",
              "always": choice == "always"}
    response = _methods["reload.mcp"]("slash-reload-mcp", params)
    if response.get("error"):
        return str(response["error"].get("message") or "MCP reload failed")
    result = response.get("result") or {}
    if result.get("status") == "confirm_required":
        return str(result.get("message") or "")
    lines = ["MCP servers reloaded" + (" (confirmation turned off for this profile)." if params["always"] else ".")]
    if failed := result.get("failed"):
        lines.append(f"Did not connect: {', '.join(failed)}")
    return "\n".join(lines)


def _format_live_reload_skills_output(sid: str, session, arg: str) -> str:
    """``slash.exec /reload-skills`` runs the ``skills.reload`` RPC in the session's own scope."""
    response = _methods["skills.reload"]("slash-reload-skills", {"session_id": sid})
    if response.get("error"):
        return str(response["error"].get("message") or "skills reload failed")
    return str((response.get("result") or {}).get("output") or "")


def register(server) -> None:
    """Publish this module's helpers onto ``server``'s globals."""
    bind_module(globals(), server)
