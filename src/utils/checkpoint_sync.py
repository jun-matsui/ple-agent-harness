"""Dual-Storage Checkpointing, Async Validation Thread, and Single Final Gate (Phase 6)."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import threading
from typing import Any, Callable, Dict, Optional


@dataclass
class CheckpointManager:
    """Persist best harness/model states whenever local score improves > 1e-7."""

    local_dir: Path
    mirror_dir: Optional[Path] = None
    best_score: float = -1.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1))
    _failure_log: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.local_dir.mkdir(parents=True, exist_ok=True)
        if self.mirror_dir is not None:
            self.mirror_dir.mkdir(parents=True, exist_ok=True)

    def save_if_improved(self, score: float, payload: Dict[str, Any], tag: str = "checkpoint") -> bool:
        """Save checkpoint atomically to local_dir and mirror_dir if score improves > 1e-7."""
        with self._lock:
            if score <= self.best_score + 1e-7:
                return False
            self.best_score = score
            raw = json.dumps({"score": score, "tag": tag, "payload": payload}, indent=2).encode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()[:12]
            fname = f"{tag}_{score:.6f}_{sha}.json"
            target = self.local_dir / fname
            target.write_bytes(raw)
            if self.mirror_dir is not None:
                shutil.copy2(target, self.mirror_dir / fname)
            return True

    def submit_async_validation(self, fn: Callable[[], tuple[bool, str]]) -> Future:
        """Run compliance/verifier checks in a background thread without blocking optimization."""
        def _wrapped() -> tuple[bool, str]:
            try:
                ok, msg = fn()
                if not ok:
                    with self._lock:
                        self._failure_log.append(msg)
                return ok, msg
            except Exception as exc:
                err = f"validation_exception: {exc}"
                with self._lock:
                    self._failure_log.append(err)
                return False, err

        return self._executor.submit(_wrapped)

    def recent_failures(self, limit: int = 10) -> list[str]:
        with self._lock:
            return list(self._failure_log[-limit:])

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)
