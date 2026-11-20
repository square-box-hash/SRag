"""Security regressions for the HTTP server.

Each test reloads `srag.server` with its own environment, because the CORS
allowlist, the bearer token and the ingest roots are all read at import time.

The vulnerability these pin down: with `allow_origins=["*"]` and no auth, any
web page a user visited could POST /ingest with a path like /home/user/notes.txt
and then POST /query to read the contents back.
"""

import importlib
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fresh_server(monkeypatch, **env):
    """Reload srag.server with a clean environment for this test."""
    for key in ("SRAG_ALLOWED_ORIGINS", "SRAG_API_TOKEN", "SRAG_INGEST_ROOTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    import srag.server as server

    return importlib.reload(server)


class FakeSRag:
    """Stands in for the real SRag so tests never load a model or an index."""

    def delete_session(self, session):
        return session == "known"

    def ingest(self, source, session=None, **kwargs):
        raise RuntimeError(f"disk on fire at {source}")


# ── CORS ────────────────────────────────────────────────────────────────────


def test_foreign_origin_is_refused(monkeypatch):
    server = fresh_server(monkeypatch)
    client = TestClient(server.app)

    res = client.get("/", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in res.headers, (
        "a third-party origin must not be allowed to read responses"
    )


def test_local_dev_origin_is_allowed(monkeypatch):
    server = fresh_server(monkeypatch)
    client = TestClient(server.app)

    res = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_wildcard_requires_explicit_opt_in(monkeypatch):
    server = fresh_server(monkeypatch, SRAG_ALLOWED_ORIGINS="*")
    client = TestClient(server.app)

    res = client.get("/", headers={"Origin": "https://anywhere.example"})
    assert res.headers.get("access-control-allow-origin") == "*"


def test_custom_allowlist_is_honoured(monkeypatch):
    server = fresh_server(
        monkeypatch, SRAG_ALLOWED_ORIGINS="https://app.example,https://other.example"
    )
    client = TestClient(server.app)

    allowed = client.get("/", headers={"Origin": "https://app.example"})
    assert allowed.headers.get("access-control-allow-origin") == "https://app.example"

    denied = client.get("/", headers={"Origin": "https://nope.example"})
    assert "access-control-allow-origin" not in denied.headers


# ── Bearer token ────────────────────────────────────────────────────────────


def test_token_is_optional_by_default(monkeypatch):
    server = fresh_server(monkeypatch)
    client = TestClient(server.app)

    assert client.get("/").status_code == 200


def test_token_is_enforced_when_set(monkeypatch):
    server = fresh_server(monkeypatch, SRAG_API_TOKEN="s3cret")

    client = TestClient(server.app)
    assert client.get("/").status_code == 401

    wrong = client.get("/", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401

    ok = client.get("/", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


def test_health_stays_open_for_liveness_probes(monkeypatch):
    server = fresh_server(monkeypatch, SRAG_API_TOKEN="s3cret")
    client = TestClient(server.app)

    assert client.get("/health").status_code == 200


def test_401_carries_cors_headers(monkeypatch):
    """The middleware order matters: a browser must see the real error."""
    server = fresh_server(
        monkeypatch, SRAG_API_TOKEN="s3cret", SRAG_ALLOWED_ORIGINS="http://localhost:3000"
    )
    client = TestClient(server.app)

    res = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert res.status_code == 401
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


# ── Ingest path confinement ─────────────────────────────────────────────────


def test_ingest_refuses_paths_outside_roots(monkeypatch, tmp_path):
    server = fresh_server(monkeypatch, SRAG_INGEST_ROOTS=str(tmp_path))
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    res = client.post("/ingest", json={"source": "/etc/secrets.txt", "session": "s"})
    assert res.status_code == 403
    assert "SRAG_INGEST_ROOTS" in res.json()["detail"]


def test_ingest_allows_paths_inside_roots(monkeypatch, tmp_path):
    server = fresh_server(monkeypatch, SRAG_INGEST_ROOTS=str(tmp_path))
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    inside = tmp_path / "notes.txt"
    inside.write_text("hello")
    res = client.post("/ingest", json={"source": str(inside), "session": "s"})
    assert res.status_code != 403, "a path inside the allowed root must not be refused"


def test_ingest_is_unrestricted_without_roots(monkeypatch):
    server = fresh_server(monkeypatch)
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    res = client.post("/ingest", json={"source": "/etc/secrets.txt", "session": "s"})
    assert res.status_code != 403, "no roots configured means unchanged behaviour"


# ── Error detail leakage ────────────────────────────────────────────────────


def test_internal_errors_do_not_leak_details(monkeypatch):
    server = fresh_server(monkeypatch)
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    res = client.post("/ingest", json={"source": "/tmp/x.txt", "session": "s"})
    assert res.status_code == 500

    detail = res.json()["detail"]
    assert "/tmp/x.txt" not in detail, "paths must not reach the client"
    assert "disk on fire" not in detail, "exception text must not reach the client"
    assert "server logs" in detail


# ── Session deletion ────────────────────────────────────────────────────────


def test_delete_reports_missing_session(monkeypatch):
    server = fresh_server(monkeypatch)
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    assert client.delete("/sessions/unknown").status_code == 404


def test_delete_reports_success(monkeypatch):
    server = fresh_server(monkeypatch)
    server.get_srag = lambda: FakeSRag()
    client = TestClient(server.app)

    res = client.delete("/sessions/known")
    assert res.status_code == 200
    assert res.json() == {"session": "known", "deleted": True}
