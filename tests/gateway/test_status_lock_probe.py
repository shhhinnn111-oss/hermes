"""An unwritable gateway runtime lock is also what a LIVE gateway under another account (root/systemd,
an elevated Windows process) looks like: only a provably stale one may be cleaned up (#42685)."""

import json
import logging
import os
from pathlib import Path

import pytest

from gateway import status


def _deny_lock_opens(monkeypatch, lock_path, *, deny_read=False):
    real_open = open

    def guarded(path, mode="r", *args, **kwargs):
        if str(path) == str(lock_path) and (deny_read or any(flag in mode for flag in "aw+")):
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded)


def _hold(lock_path):
    holder = open(lock_path, "a+", encoding="utf-8")
    assert status._try_acquire_file_lock(holder)
    return holder


def _home_lock(tmp_path, monkeypatch, content="x"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    lock_path = status._get_gateway_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(content, encoding="utf-8")
    return lock_path


@pytest.mark.platforms("any")
class TestUnwritableRuntimeLock:
    def test_held_lock_reports_active_and_is_kept(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("x", encoding="utf-8")
        holder = _hold(lock_path)
        try:
            _deny_lock_opens(monkeypatch, lock_path)
            assert status.is_gateway_runtime_lock_active(lock_path) is True
            assert lock_path.exists()
        finally:
            status._release_file_lock(holder)
            holder.close()

    def test_unreadable_lock_with_live_pid_record_is_kept(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("x", encoding="utf-8")
        (tmp_path / "gateway.pid").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        assert status.is_gateway_runtime_lock_active(lock_path) is True
        assert lock_path.exists()

    def test_unreadable_lock_with_dead_pid_record_is_stale(self, tmp_path, monkeypatch):
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("stale", encoding="utf-8")
        (tmp_path / "gateway.pid").write_text(json.dumps({"pid": 999999}), encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        assert status.is_gateway_runtime_lock_active(lock_path) is False
        assert not lock_path.exists()

    def test_unreadable_lock_without_pid_record_is_kept(self, tmp_path, monkeypatch):
        """A gateway takes the lock before publishing gateway.pid, and a 0600 pair under
        another account is unreadable: missing metadata proves nothing, so keep the lock."""
        lock_path = tmp_path / "gateway.lock"
        lock_path.write_text("x", encoding="utf-8")
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        assert status.is_gateway_runtime_lock_active(lock_path) is True
        assert lock_path.exists()

    def test_acquire_refuses_a_held_lock(self, tmp_path, monkeypatch):
        lock_path = _home_lock(tmp_path, monkeypatch)
        holder = _hold(lock_path)
        try:
            _deny_lock_opens(monkeypatch, lock_path)
            assert status.acquire_gateway_runtime_lock() is False
            assert lock_path.exists()
        finally:
            status._release_file_lock(holder)
            holder.close()
            status.release_gateway_runtime_lock()

    def test_acquire_refuses_an_uninspectable_lock_and_says_why(self, tmp_path, monkeypatch, caplog):
        lock_path = _home_lock(tmp_path, monkeypatch)
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        with caplog.at_level(logging.WARNING, logger=status.logger.name):
            assert status.acquire_gateway_runtime_lock() is False
        assert lock_path.exists()
        assert "cannot be inspected" in caplog.text

    @pytest.mark.parametrize("scoped", [False, True])
    def test_get_running_pid_cleanup_keeps_an_uninspectable_lock(self, tmp_path, monkeypatch, scoped):
        lock_path = _home_lock(tmp_path, monkeypatch)
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        assert status.get_running_pid(tmp_path / "gateway.pid" if scoped else None) is None
        assert lock_path.exists()

    def test_startup_claim_refuses_and_keeps_an_uninspectable_lock(self, tmp_path, monkeypatch):
        """Startup polls get_running_pid() before it acquires: neither step may delete the lock."""
        from gateway.run import _start_gateway_claim_pid_file

        lock_path = _home_lock(tmp_path, monkeypatch)
        _deny_lock_opens(monkeypatch, lock_path, deny_read=True)
        assert _start_gateway_claim_pid_file() is False
        assert lock_path.exists()
        assert not (tmp_path / "gateway.pid").exists()

    def test_undeletable_stale_lock_keeps_its_proof_until_it_can_be_replaced(self, tmp_path, monkeypatch):
        """If unlinking a provably stale lock fails, cleanup must keep the dead gateway.pid that
        proves it stale (else every later start sees an uninspectable lock and refuses), and the
        start that finally replaces the lock must clear that record before its O_EXCL pid write."""
        lock_path = _home_lock(tmp_path, monkeypatch, "stale")
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({"pid": 999999}), encoding="utf-8")
        monkeypatch.setattr(status, "_pid_exists", lambda pid: False)
        foreign = {"lock": True, "failures_left": 1}  # fails in the poll, succeeds in acquire
        real_open, real_unlink = open, Path.unlink

        def guarded_open(path, mode="r", *args, **kwargs):
            if foreign["lock"] and str(path) == str(lock_path):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, mode, *args, **kwargs)

        def flaky_unlink(path, *args, **kwargs):
            if str(path) == str(lock_path):
                if foreign["failures_left"] > 0:
                    foreign["failures_left"] -= 1
                    raise PermissionError(13, "Permission denied", str(path))
                foreign["lock"] = False
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", guarded_open)
        monkeypatch.setattr(Path, "unlink", flaky_unlink)
        # The startup claim order: poll, acquire, then the exclusive pid write.
        assert status.get_running_pid() is None
        assert lock_path.exists() and pid_path.exists()
        try:
            assert status.acquire_gateway_runtime_lock() is True
            status.write_pid_file()
            assert json.loads(pid_path.read_text(encoding="utf-8"))["pid"] == os.getpid()
        finally:
            status.release_gateway_runtime_lock()

    def test_replacing_a_stale_lock_keeps_a_pid_file_it_cannot_prove_dead(self, tmp_path, monkeypatch):
        """A racing starter's gateway.pid is empty between its O_EXCL create and its write: only a
        readable dead record may be dropped, so the exclusive pid write still picks one winner."""
        lock_path = _home_lock(tmp_path, monkeypatch, "stale")
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text("", encoding="utf-8")
        foreign = {"lock": True}
        real_open, real_unlink = open, Path.unlink

        def guarded_open(path, mode="r", *args, **kwargs):
            if foreign["lock"] and str(path) == str(lock_path) and any(flag in mode for flag in "aw+"):
                raise PermissionError(13, "Permission denied", str(path))
            return real_open(path, mode, *args, **kwargs)

        def unlink(path, *args, **kwargs):
            if str(path) == str(lock_path):
                foreign["lock"] = False
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", guarded_open)
        monkeypatch.setattr(Path, "unlink", unlink)
        try:
            assert status.acquire_gateway_runtime_lock() is True
            assert pid_path.exists()
            with pytest.raises(FileExistsError):
                status.write_pid_file()
        finally:
            status.release_gateway_runtime_lock()

    def test_get_running_pid_keeps_an_unreadable_pid_file_beside_a_held_lock(self, tmp_path, monkeypatch):
        """A live gateway under another account may own a 0600 gateway.pid: unreadable is not stale."""
        lock_path = _home_lock(tmp_path, monkeypatch)
        pid_path = tmp_path / "gateway.pid"
        pid_path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
        holder = _hold(lock_path)
        try:
            _deny_lock_opens(monkeypatch, lock_path)
            real_read_text = Path.read_text

            def unreadable_pid(path, *args, **kwargs):
                if str(path) == str(pid_path):
                    raise PermissionError(13, "Permission denied", str(path))
                return real_read_text(path, *args, **kwargs)

            monkeypatch.setattr(Path, "read_text", unreadable_pid)
            assert status.get_running_pid() is None
            assert lock_path.exists() and pid_path.exists()
        finally:
            status._release_file_lock(holder)
            holder.close()


@pytest.mark.platforms("windows")
def test_empty_unheld_lock_is_stale_on_windows(tmp_path, monkeypatch):
    """The read-only probe must not try to pad an empty lock (the write would fail and read as
    contention); Windows allows locking past EOF."""
    lock_path = tmp_path / "gateway.lock"
    lock_path.write_text("", encoding="utf-8")
    _deny_lock_opens(monkeypatch, lock_path)
    assert status.is_gateway_runtime_lock_active(lock_path) is False
    assert not lock_path.exists()
