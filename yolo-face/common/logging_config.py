"""
Shared logging configuration for all microservices.

Emits JSON-structured lines to stdout so Docker/Compose log drivers and
log aggregators (Loki, Fluentd, CloudWatch) can parse them without extra
configuration.  Falls back to a human-readable format when LOG_JSON=0.
"""

import logging
import os
import sys

_configured = False

LOG_JSON = os.environ.get("LOG_JSON", "1") == "1"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


class _JsonFormatter(logging.Formatter):
    """Minimal single-line JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    """Configure root logger once; subsequent calls are no-ops."""
    global _configured
    if _configured:
        return
    _configured = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _JsonFormatter() if LOG_JSON else logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, ensuring root is configured first."""
    configure_logging()
    return logging.getLogger(name)
