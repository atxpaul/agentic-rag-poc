import json
from typing import Dict

from . import config


def _parse_json_env(raw: str, fallback):
    if not raw:
        return fallback
    try:
        val = json.loads(raw)
        return val if isinstance(val, (dict, list)) else fallback
    except Exception:
        return fallback


PROMPTS_BY_DOMAIN: Dict[str, str] = _parse_json_env(
    config.PROMPT_SYSTEM_BY_DOMAIN_RAW, {
        "default": config.PROMPT_SYSTEM_GENERIC}
)
if "default" not in PROMPTS_BY_DOMAIN:
    PROMPTS_BY_DOMAIN["default"] = config.PROMPT_SYSTEM_GENERIC


def select_system_prompt(domain: str) -> str:
    return PROMPTS_BY_DOMAIN.get(domain) or PROMPTS_BY_DOMAIN["default"]


def verifier_prompts():
    return config.VERIFY_PROMPT_SYSTEM, config.VERIFY_PROMPT_HUMAN
