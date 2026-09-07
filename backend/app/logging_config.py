import json
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_FILE = LOG_DIR / "audit.jsonl"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
        }

        for key in [
            "api_key",
            "image_id",
            "original_filename",
            "size",
            "sha256",
            "mime_type",
            "task_type",
            "models_used",
            "reason",
            "client_ip",
        ]:
            value = getattr(record, key, None)

            if value is not None:
                log_record[key] = value

        return json.dumps(log_record)


audit_logger = logging.getLogger("satquery_audit")
audit_logger.setLevel(logging.INFO)
audit_logger.propagate = False

if not audit_logger.handlers:
    formatter = JsonFormatter()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        AUDIT_LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    audit_logger.addHandler(console_handler)
    audit_logger.addHandler(file_handler)


def log_event(event: str, **kwargs):
    audit_logger.info(
        event,
        extra={
            "event": event,
            **kwargs
        },
    )