"""Verify audit mappings against actual redaction output, without Redis."""
import json

import pytest

from llm_shield_proxy.engines import hits_recorder
from llm_shield_proxy.engines.pii_engine import PIIEngine
from llm_shield_proxy.engines.vault import Vault


@pytest.mark.parametrize("synthetic", [False, True])
def test_records_actual_replacement(monkeypatch, synthetic):
    vault = Vault(synthetic=synthetic)
    engine = PIIEngine(enable_tier2=False, enable_tier3=False)
    sent = []

    class Redis:
        def setex(self, key, ttl, payload):
            sent.append((key, json.loads(payload)))

    monkeypatch.setattr(hits_recorder, "_redis_client", lambda: Redis())
    hits_recorder.clear_hits(vault)
    original = "john.doe@acme.org"
    output = engine.redact_text(f"Email {original}; again {original}", vault)
    hits_recorder.flush_hits(vault, "request-one")
    assert original not in output
    assert sent[0][0] == "redact_hits:request-one"
    mapping = [h for h in sent[0][1] if h["value"] == original]
    assert mapping
    for hit in mapping:
        assert hit["type"] == "EMAIL"
        assert hit["replacement"] != original
        assert hit["replacement"] in output
    hits_recorder.clear_hits(vault)
    hits_recorder.flush_hits(vault, "request-two")
    assert len(sent) == 1


def test_preserves_multiple_replacements(monkeypatch):
    vault = Vault(synthetic=False)
    sent = []

    class Redis:
        def setex(self, key, ttl, payload):
            sent.extend(json.loads(payload))

    monkeypatch.setattr(hits_recorder, "_redis_client", lambda: Redis())
    hits_recorder.record_hit(vault, "USERNAME", "moon_dev", "Nathan161")
    hits_recorder.record_hit(vault, "USERNAME", "moon_dev", "Laura618")
    hits_recorder.record_hit(vault, "USERNAME", "moon_dev", "Nathan161")
    hits_recorder.flush_hits(vault, "request-one")
    assert len(sent) == 2
    assert {h["replacement"] for h in sent} == {"Nathan161", "Laura618"}
