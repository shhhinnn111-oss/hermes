"""An OAuth server parked on an authentication failure logs "parking until credentials change".
The timed self-probe must honour that: with the same token/client files on disk it can only
replay the dead refresh token (a 400 at the token endpoint) and fail the non-interactive
authorize step again, every ``_PARKED_RETRY_INTERVAL``. A new token file (``hermes mcp login``
from another process) or an explicit reconnect must still revive the server."""

import asyncio

import pytest


@pytest.mark.no_isolate
def test_auth_parked_oauth_server_probes_only_after_credentials_change(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools import mcp_tool_config as _config
    from tools.mcp_oauth import HermesTokenStorage, OAuthNonInteractiveError
    from tools.mcp_tool import MCPServerTask

    monkeypatch.setattr(mcp_tool, "_PARKED_RETRY_INTERVAL", 0.02)
    server_cfg = {"url": "https://mcp.example.com/mcp", "auth": "oauth"}
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"srv": dict(server_cfg)})

    tokens = HermesTokenStorage("srv")._tokens_path()
    tokens.parent.mkdir(parents=True, exist_ok=True)
    tokens.write_text('{"access_token": "A1", "token_type": "Bearer", "refresh_token": "R1"}')

    state = {"transport_calls": 0, "deregistered": 0, "probe_wakes": 0}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return True

            async def _prepare_run(self, config):
                self._config = config
                self._auth_type = "oauth"
                return True

            def _deregister_tools(self):
                state["deregistered"] += 1

            async def _wait_for_reconnect_or_shutdown(self, timeout):
                outcome = await super()._wait_for_reconnect_or_shutdown(timeout)
                state["probe_wakes"] += outcome == "self-probe"
                return outcome

            async def _run_http(self, config):
                state["transport_calls"] += 1
                raise OAuthNonInteractiveError("MCP OAuth requires browser authorization")

        async def _wait_for(pred, rounds=1000):
            for _ in range(rounds):
                if pred():
                    return True
                await asyncio.sleep(0.01)
            return pred()

        async def _sleep_through_probes(count=5):
            target = state["probe_wakes"] + count
            assert await _wait_for(lambda: state["probe_wakes"] >= target), "self-probe never woke"

        task = _Task("srv")
        run_task = asyncio.ensure_future(task.run(dict(server_cfg)))

        assert await _wait_for(lambda: state["deregistered"] >= 1), "server never parked"
        assert not run_task.done(), "run task exited instead of parking"

        parked_at = state["transport_calls"]
        await _sleep_through_probes()
        assert state["transport_calls"] == parked_at, (
            "auth-parked self-probe retried with unchanged credentials "
            f"({state['transport_calls'] - parked_at} extra attempts)")

        # Re-authorization from another process writes a new token file: the next probe revives.
        tokens.write_text('{"access_token": "A2", "token_type": "Bearer", "refresh_token": "R2-new"}')
        assert await _wait_for(lambda: state["transport_calls"] > parked_at), (
            "self-probe did not revive after the token file changed")

        # Still failing: re-parked against the NEW snapshot, and quiet again.
        assert await _wait_for(lambda: state["deregistered"] >= 2), "server never re-parked"
        settled = state["transport_calls"]
        await _sleep_through_probes()
        assert state["transport_calls"] == settled

        # An explicit reconnect (dashboard re-auth, manual refresh) is never gated.
        task._reconnect_event.set()
        assert await _wait_for(lambda: state["transport_calls"] > settled), (
            "explicit reconnect was gated by the credential snapshot")

        task._shutdown_event.set()
        task._reconnect_event.set()
        await asyncio.wait_for(run_task, timeout=15)

    asyncio.run(_scenario())
