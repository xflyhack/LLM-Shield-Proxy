"""Session-scoped memoization of the pure half of redact_text.

What is cached and why it is safe
---------------------------------
`redact_text` runs two pure steps on every string -- normalize_and_desmuggle
(NFKC + de-smuggling) and `detect_spans` (the whole Tier 1/2/3 cascade). Their
result depends only on (text, active_profile), never on vault state. Minting
depends on vault state and stays uncached. A multi-turn client replays its
whole history every request, so without a cache the engine re-scans the same
strings -- the most expensive ones included -- on every turn.

Why the Vault object
--------------------
Vault has no __slots__, so the cache hangs directly on the instance via
setattr: no official file is restructured to hold it. A vault lives exactly as
long as its session, and VaultStore LRU/TTL eviction and Redis restore hand
out fresh instances, so the cache inherits the session lifecycle. Custom
regex profiles load once at container start (`_init_custom_regex` has no hot
reload): a rule change means a restart, which drops every vault and every
cache with it. That is the whole invalidation story -- there is nothing else
to get wrong. (`dump_state`/`load_state` whitelist their fields, so the cache
never travels through Redis either.)

Profile variants
----------------
Profiles are stable objects, but the API layer wraps them per request (for
example a Tier-3-stripped copy), so object identity is not a stable key. The
fingerprint below separates every variant one vault can meet. If some future
variant collided on all three fields, entries would merely be missed, never
wrong: detector configuration only changes on restart, which clears the
cache anyway.

Concurrency
-----------
redact_payload runs in a thread pool and concurrent requests can share one
session vault. No lock: single dict operations are atomic under the GIL, so
the worst interleaving computes the same entry twice, and one thread's
accounting drift only makes eviction fire early or late. The lock on the
vault itself is NOT reused -- minting holds it and would serialize the hot
path.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional, Tuple

# One session replays a bounded string set (history, system prompt, tool
# definitions). The caps only stop pathological sessions from growing the
# cache forever; normal traffic sits far below them.
_MAX_CACHED_TEXT_CHARS = 1024 * 1024  # a single larger leaf is computed, not stored
_MAX_CACHE_ENTRIES = 4096
_MAX_CACHE_CHARS = 8 * 1024 * 1024  # text + working_text (+ a flat span allowance)

_VAULT_ATTR = "_span_cache"

# (profile name, tier1 pattern count, sorted tier3 entity types)
_Fingerprint = Tuple[str, int, Tuple[str, ...]]


def _fingerprint(profile: Any) -> _Fingerprint:
    """A cheap, stable identity for the profile variant the vault is seeing."""
    if profile is None:
        return ("__default__", 0, ())
    tier1 = getattr(profile, "tier1_patterns", None) or ()
    tier3 = getattr(profile, "tier3_ner_entities", None) or set()
    return (str(getattr(profile, "name", "")), len(tier1), tuple(sorted(tier3)))


class _SpanCache:
    """Flat LRU of (fingerprint, text) -> (working_text, spans)."""

    __slots__ = ("_entries", "_chars")

    def __init__(self) -> None:
        self._entries: "OrderedDict[Tuple[_Fingerprint, str], Tuple[str, Any]]" = OrderedDict()
        self._chars = 0

    def get(self, key: Tuple[_Fingerprint, str]) -> Optional[Tuple[str, Any]]:
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)  # LRU: replayed history stays resident
        return entry

    def put(self, key: Tuple[_Fingerprint, str], entry: Tuple[str, Any]) -> None:
        text = key[1]
        if len(text) > _MAX_CACHED_TEXT_CHARS:
            return
        working_text, spans = entry
        old = self._entries.pop(key, None)
        if old is not None:
            self._chars -= len(key[1]) + len(old[0]) + 64 * len(old[1])
        self._entries[key] = entry
        self._chars += len(text) + len(working_text) + 64 * len(spans)
        while (self._chars > _MAX_CACHE_CHARS or len(self._entries) > _MAX_CACHE_ENTRIES) \
                and self._entries:
            try:
                evicted_key, evicted = self._entries.popitem(last=False)
            except KeyError:  # a concurrent writer emptied it mid-loop
                break
            self._chars -= len(evicted_key[1]) + len(evicted[0]) + 64 * len(evicted[1])


def cached_spans_for(
    engine: Any,
    vault: Any,
    profile: Any,
    text: str,
) -> Tuple[str, Any]:
    """(working_text, spans) for `text` under `profile`, memoized on the vault.

    The single integration point called from `pii_engine.redact_text`: on a
    hit it is one dict read; on a miss it computes exactly what the uncached
    path computed and remembers it. Vaults that refuse new attributes (none
    exist today) simply run uncached.
    """
    cache = getattr(vault, _VAULT_ATTR, None)
    if cache is None:
        try:
            cache = _SpanCache()
            setattr(vault, _VAULT_ATTR, cache)
        except AttributeError:
            return _compute(engine, profile, text)
    key = (_fingerprint(profile), text)
    entry = cache.get(key)
    if entry is None:
        entry = _compute(engine, profile, text)
        cache.put(key, entry)
    return entry


def _compute(engine: Any, profile: Any, text: str) -> Tuple[str, Any]:
    # Imported lazily: pii_engine imports this module at load time, so a
    # top-level import back would be circular. By the time this runs,
    # pii_engine is fully loaded.
    from llm_shield_proxy.engines.pii_engine import normalize_and_desmuggle

    working_text = normalize_and_desmuggle(text)
    return working_text, engine.detect_spans(working_text, profile)
