"""Board access with an OpenShell Sandbox Passport instead of a bearer token.

Spawn Gate registers each sandbox (team, agent, role) through the admin endpoint; the
board then trusts the Passport attached outside the agent and locks the caller to its
team's space."""

import base64
import json
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from coworker.teams.model import Actor, Role

AUD = "host.openshell.internal:8765"
ADMIN = {"X-Board-Admin": "admin-secret"}
LEAD = Actor("user", Role.USER)


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def passport(key, sandbox_id, audience=AUD, iat=None, issuer="openshell-teams/passport"):
    now = int(iat if iat is not None else time.time())
    header = _b64(json.dumps({"alg": "EdDSA", "typ": "JWT"}).encode())
    claims = _b64(json.dumps({"iss": issuer, "aud": audience, "sbx": sandbox_id, "name": "x",
                              "iat": now, "exp": now + 60}).encode())
    sig = _b64(key.sign(f"{header}.{claims}".encode()))
    return {"X-OpenShell-Caller": f"{header}.{claims}.{sig}"}


@pytest.fixture
def api(tmp_path, monkeypatch):
    key = Ed25519PrivateKey.generate()
    pub = tmp_path / "passport.pub"
    pub.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM,
                                                  serialization.PublicFormat.SubjectPublicKeyInfo))
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("COWORKER_API_TOKEN", "sidecar-secret")
    monkeypatch.setenv("OPENWORKER_PASSPORT_PUBLIC_KEY", str(pub))
    monkeypatch.setenv("OPENWORKER_PASSPORT_AUDIENCE", AUD)
    monkeypatch.setenv("OPENWORKER_BOARD_ADMIN_TOKEN", "admin-secret")
    from fastapi.testclient import TestClient

    from coworker.permissions import Mode
    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    manager = SessionManager(workspace=None, data_dir=tmp_path / "state",
                             model="openai:gpt-test", mode=Mode("interactive"))
    client = TestClient(create_app(manager), base_url="http://board.test")
    client.post("/v1/admin/agents", headers=ADMIN,
                json={"sandbox_id": "sb-rev", "team": "proj", "name": "reviewer", "role": "worker"})
    client.post("/v1/admin/agents", headers=ADMIN,
                json={"sandbox_id": "sb-lead", "team": "proj", "name": "lead", "role": "lead"})
    yield client, manager, key
    client.close()


def test_whoami_comes_from_the_registration(api):
    client, _, key = api
    r = client.get("/v1/board/whoami", headers=passport(key, "sb-rev"))
    assert r.status_code == 200 and r.json() == {"actor": "reviewer", "role": "worker"}


def test_comment_is_attributed_to_the_registered_agent(api):
    client, manager, key = api
    item = manager.team_store.create_item("proj", LEAD, title="Fix", criteria="tests pass")
    manager.team_store.assign("proj", LEAD, item["id"], "reviewer")
    r = client.post("/v1/board/items/comment", headers=passport(key, "sb-rev"),
                    json={"space": "proj", "id": item["id"], "body": "LGTM", "author": "lead"})
    assert r.status_code == 200, r.text
    comments = client.get("/v1/board/comments", headers=passport(key, "sb-rev"),
                          params={"space": "proj", "id": item["id"]}).json()
    [entry] = [c for c in comments["comments"] if c["body"] == "LGTM"]
    assert entry["author"] == "reviewer" and entry["role"] == "worker"


def test_role_rules_still_apply(api):
    client, manager, key = api
    item = manager.team_store.create_item("proj", LEAD, title="Fix", criteria="tests pass")
    r = client.post("/v1/board/items/assign", headers=passport(key, "sb-rev"),
                    json={"space": "proj", "id": item["id"], "assignee": "reviewer"})
    assert r.status_code == 403


def test_another_teams_space_is_refused(api):
    client, manager, key = api
    manager.team_store.create_item("other-team", LEAD, title="Secret", criteria="x")
    assert client.get("/v1/board/items", headers=passport(key, "sb-lead"),
                      params={"space": "other-team"}).status_code == 403
    r = client.post("/v1/board/items/comment", headers=passport(key, "sb-lead"),
                    json={"space": "other-team", "id": 1, "body": "hi"})
    assert r.status_code == 403


def test_spaces_list_shows_only_the_callers_team(api):
    client, manager, key = api
    manager.team_store.create_item("other-team", LEAD, title="Secret", criteria="x")
    manager.team_store.create_item("proj", LEAD, title="Ours", criteria="x")
    r = client.get("/v1/board/spaces", headers=passport(key, "sb-lead"))
    assert r.status_code == 200 and r.json()["spaces"] == ["proj"]


def test_unregistered_sandbox_is_refused(api):
    client, _, key = api
    assert client.get("/v1/board/whoami", headers=passport(key, "sb-ghost")).status_code == 403


def test_deregistered_sandbox_loses_access(api):
    client, _, key = api
    client.delete("/v1/admin/agents/sb-rev", headers=ADMIN)
    assert client.get("/v1/board/whoami", headers=passport(key, "sb-rev")).status_code == 403


@pytest.mark.parametrize("bad", [
    {"audience": "host.openshell.internal:8766"},
    {"iat": time.time() - 3600},
    {"issuer": "someone-else"},
])
def test_invalid_passport_is_refused_and_does_not_fall_back_to_tokens(api, bad):
    client, manager, key = api
    token = manager.board_tokens.mint("reviewer", "worker")
    headers = {**passport(key, "sb-rev", **bad), "Authorization": f"Bearer {token}"}
    assert client.get("/v1/board/whoami", headers=headers).status_code == 401


def test_passport_signed_by_another_key_is_refused(api):
    client, _, _ = api
    assert client.get("/v1/board/whoami",
                      headers=passport(Ed25519PrivateKey.generate(), "sb-rev")).status_code == 401


def test_registration_needs_the_admin_token(api):
    client, _, _ = api
    r = client.post("/v1/admin/agents",
                    json={"sandbox_id": "x", "team": "proj", "name": "x", "role": "lead"})
    assert r.status_code == 403


def test_bearer_tokens_still_work_for_clients_outside_openshell(api):
    client, manager, _ = api
    token = manager.board_tokens.mint("ext", "worker")
    r = client.get("/v1/board/whoami", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["actor"] == "ext"
