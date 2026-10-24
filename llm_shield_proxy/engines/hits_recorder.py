"""Per-request record of what redaction replaced, shipped to Redis for the
audit proxy's egress logger.

redact_text holds the exact original value of every entity it replaces --
first-hand fact, no inference. This module parks those values on the session
vault (setattr sidecar; Vault has no __slots__) while the payload is being
redacted, then flushes them to Redis keyed by the request id the ingress
proxy already stamped on the request. The audit egress reads that key when it
writes its request_log row.

Nothing here touches the forwarded payload: the data travels on a side
channel, never through the request, so it cannot leak to the provider. A
write failure logs and is forgotten -- the audit side treats a missing key as
"no data", forwarding is unaffected.

Requires REDIS_URL (the same setting the Redis vault store uses). Without it
this module records nothing and the audit side simply finds no key.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional, Set, Tuple

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except ImportError:  # pragma: no cover - deployed images always include it
    redis = None

_VAULT_ATTR = "_redact_hits"
_KEY_PREFIX = "redact_hits:"
_TTL_SECONDS = 3600  # egress writes its row within seconds; 1h is generous

_client: Any = None
_client_ready = False


def record_hit(vault: Any, entity_type: str, matched_text: str) -> None:
    """Park one replaced value on the vault sidecar. Hot path: never raises."""
    hits = getattr(vault, _VAULT_ATTR, None)
    if hits is None:
        try:
            hits = set()
            setattr(vault, _VAULT_ATTR, hits)
        except AttributeError:  # exotic vault with __slots__: skip recording
            return
    hits.add((entity_type, matched_text))


def clear_hits(vault: Any) -> None:
    """Start of request: the sidecar must hold only this request's entities.

    The vault is session-scoped and shared across requests, so leftover
    entities from an earlier turn would pollute this request's record.
    """
    try:
        setattr(vault, _VAULT_ATTR, set())
    except AttributeError:
        pass


def flush_hits(vault: Any, request_id: Optional[str]) -> None:
    """Ship the sidecar to Redis. Run in the executor; never raises."""
    hits: Optional[Set[Tuple[str, str]]] = getattr(vault, _VAULT_ATTR, None)
    if not hits or not request_id:
        return
    client = _redis_client()
    if client is None:
        return
    try:
        payload = [{"type": t, "value": v} for t, v in sorted(hits)]
        client.setex(
            f"{_KEY_PREFIX}{request_id}",
            _TTL_SECONDS,
            json.dumps(payload, ensure_ascii=False),
        )
    except Exception as exc:  # noqa: BLE001 - side channel never breaks a request
        logger.warning("redact hits flush failed: %s", exc)


def _redis_client() -> Any:
    """Lazy module-level sync client built from settings.REDIS_URL."""
    global _client, _client_ready
    if _client_ready:
        return _client
    _client_ready = True
    if redis is None:
        logger.warning("redis package missing; redact hits recording disabled")
        return None
    # Imported lazily so this module stays importable without the settings
    # stack (unit tests stub it); the call path always has it loaded.
    from llm_shield_proxy.core.config import settings

    if not settings.REDIS_URL:
        logger.warning("REDIS_URL unset; redact hits recording disabled")
        return None
    # Socket timeouts are mandatory: this flush runs in the default executor,
    # the same pool that runs payload redaction on the hot path. Against a
    # black-holed Redis (accepts nothing, answers nothing) an unbounded setex
    # would pin a worker thread forever, and enough of those stall redaction
    # itself. The payload here is tiny; 2s is generous.
    _client = redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    return _client
