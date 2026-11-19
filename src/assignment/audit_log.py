"""
Assignment 11 — Audit Log.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from guardrails.output_guardrails import content_filter


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[tuple[str, str | None], tuple[str, str, float]] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Start a timed entry and redact the prompt before storing it."""
        started = time.monotonic()
        self._open[user_id, request_id] = (
            content_filter(text)["redacted"], utc_now_iso(), started
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store the layer decision, redacted reply and elapsed time."""
        pending = self._open.pop((user_id, request_id), None)
        now = time.monotonic()
        prompt, timestamp, started = pending if pending else (None, utc_now_iso(), now)
        self.logs.append({
            "user_id": user_id,
            "request_id": request_id,
            "timestamp": timestamp,
            "input": prompt,
            "output": content_filter(text)["redacted"],
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((now - started) * 1000, 3),
        })

    def export_json(self, filepath: str | Path | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
