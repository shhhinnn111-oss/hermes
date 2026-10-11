"""Probing a gateway runtime lock this process does not own: strict probes, unwritable locks.

An unwritable ``gateway.lock`` (owned by root / another account) is what a stale lock left by a root
launchd session looks like (#42685), but it is also what a LIVE gateway under another account looks
like, so only a provably stale one may be removed.
"""

from __future__ import annotations

import contextlib
import errno
import sys
from pathlib import Path
from typing import Literal, Optional

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def _strict_path_exists(path: Path, label: str) -> bool:
    """Like ``path.exists()`` but raises RuntimeError instead of False on EACCES-style errors."""
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"{label} metadata is not inspectable: {exc}") from exc


def _is_gateway_runtime_lock_active_strict(lock_path: Path) -> bool:
    """Probe ownership without treating access failures as absence."""
    from gateway.status import _probe_lock_file

    try:
        handle = open(lock_path, "r+", encoding="utf-8")
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock is not inspectable: {exc}") from exc
    try:
        return _probe_lock_file(handle)
    except OSError as exc:
        raise RuntimeError(f"gateway runtime lock probe failed: {exc}") from exc


def probe_lock_readonly(lock_path: Path) -> Optional[bool]:
    """Read-only lock probe: True held, False unheld, None when it cannot be determined.

    Never writes (an unwritable lock must not be padded), and on POSIX takes a shared lock,
    which conflicts with the gateway's exclusive one but, unlike LOCK_EX, does not need write
    access on NFS. Only lock contention counts as "held"; any other failure is "unknown".
    """
    from gateway.status import _WINDOWS_LOCK_OFFSET

    try:
        handle = open(lock_path, "rb")
    except OSError:
        return None
    try:
        if sys.platform == "win32":
            # Locking past EOF is allowed, so an empty file needs no padding byte.
            handle.seek(_WINDOWS_LOCK_OFFSET)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                return True if exc.errno in (errno.EACCES, errno.EDEADLK) else None
            with contextlib.suppress(OSError):
                handle.seek(_WINDOWS_LOCK_OFFSET)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            return False
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return None
        with contextlib.suppress(OSError):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return False
    finally:
        with contextlib.suppress(OSError):
            handle.close()


def classify_unwritable_lock(lock_path: Path) -> Literal["held", "stale", "unknown"]:
    """State of a runtime lock this process cannot open for writing.

    Prefer the lock itself (read-only probe). If even that is impossible, fall back to the
    sibling gateway.pid: a readable record naming a dead process proves the lock stale, a live
    one proves it held. Missing or unreadable metadata proves nothing: a gateway acquires the
    lock before it publishes gateway.pid, and a 0600 pair under another account is unreadable.
    """
    from gateway.status import _live_pid_from_record, _pid_from_record, _read_pid_record

    probe = probe_lock_readonly(lock_path)
    if probe is not None:
        return "held" if probe else "stale"
    pid_record = _read_pid_record(lock_path.with_name("gateway.pid"))
    if _pid_from_record(pid_record) is None:
        return "unknown"
    return "held" if _live_pid_from_record(pid_record) is not None else "stale"


def lock_is_unwritable(lock_path: Path) -> bool:
    """True when the lock exists but this process may not open it for writing."""
    try:
        open(lock_path, "rb+").close()
    except PermissionError:
        return True
    except OSError:
        return False
    return False
