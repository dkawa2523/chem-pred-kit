from __future__ import annotations

import os
from typing import Any, Dict, Optional


def resolve_api_key(cfg: Dict[str, Any], required: bool = False) -> Optional[str]:
    if cfg.get("api_key"):
        raise ValueError("api_key must be provided via environment variables; set api_key_env instead.")
    env_name = cfg.get("api_key_env")
    if not env_name:
        return None
    api_key = os.environ.get(str(env_name))
    if required and not api_key:
        raise ValueError(f"Environment variable '{env_name}' is required but not set.")
    return api_key
