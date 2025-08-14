import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from . import config


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("rag")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_path = Path(config.RAG_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


_LOGGER = setup_logger()


def log_event(event: str, data: Dict[str, Any]) -> None:
    payload = {"ts": datetime.utcnow().isoformat() + "Z", "event": event}
    payload.update(data)
    try:
        _LOGGER.info(json.dumps(payload, ensure_ascii=False))
    except Exception:
        pass
