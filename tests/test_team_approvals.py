import time

import pytest

from coworker.teams import approvals as ap
from coworker.teams.approvals import ApprovalError, ApprovalStore, worker_digest

POLICY = {"version": 1, "network_policies": {"inference": {"endpoints": [{"host": "x", "port": 443}]}}}
REVIEWER = {"name": "reviewer", "persona": "reviewer", "role": "worker", "policy": POLICY, "providers": ["p"]}


@pytest.fixture
def store(tmp_path):
    return ApprovalStore(tmp_path / "approvals.db")


def approved(store):
    p = store.propose("proj", "lead", [REVIEWER])
    return store.decide(p["id"], approve=True, by="user")


def test_no_approval_ids_until_the_user_approves(store):
    p = store.propose("proj", "lead", [REVIEWER])
    assert p["state"] == "pending" and "approvals" not in p


def test_approval_issues_one_id_per_worker_that_works_once(store):
    p = approved(store)
    aid = p["approvals"]["reviewer"]
    store.consume(aid, space="proj", lead="lead", worker="reviewer", digest=worker_digest(REVIEWER))
    with pytest.raises(ApprovalError, match="already used"):
        store.consume(aid, space="proj", lead="lead", worker="reviewer", digest=worker_digest(REVIEWER))


@pytest.mark.parametrize("change,match", [
    ({"space": "other"}, "another team"),
    ({"lead": "lead-2"}, "another lead"),
    ({"worker": "patcher"}, "another worker"),
    ({"digest": worker_digest({**REVIEWER, "policy": {"version": 1}})}, "differs"),
])
def test_anything_different_from_the_approval_is_refused(store, change, match):
    aid = approved(store)["approvals"]["reviewer"]
    args = {"space": "proj", "lead": "lead", "worker": "reviewer", "digest": worker_digest(REVIEWER), **change}
    with pytest.raises(ApprovalError, match=match):
        store.consume(aid, **args)


def test_expired_approval_is_refused(store, monkeypatch):
    aid = approved(store)["approvals"]["reviewer"]
    later = time.time() + ap.APPROVAL_LIFETIME_SECONDS + 5
    monkeypatch.setattr(ap.time, "time", lambda: later)
    with pytest.raises(ApprovalError, match="expired"):
        store.consume(aid, space="proj", lead="lead", worker="reviewer", digest=worker_digest(REVIEWER))


def test_rejected_proposal_issues_nothing(store):
    p = store.propose("proj", "lead", [REVIEWER])
    assert store.decide(p["id"], approve=False, by="user")["state"] == "rejected"
    with pytest.raises(ApprovalError):
        store.decide(p["id"], approve=True, by="user")


# --------------------------------------------------------------- over HTTP

from test_board_passport import ADMIN, passport  # noqa: E402
from test_board_passport import api  # noqa: E402,F401


def test_lead_proposes_user_approves_spawn_gate_consumes(api):  # noqa: F811
    client, _, key = api
    lead = passport(key, "sb-lead")
    p = client.post("/v1/board/team-proposals", headers=lead, json={"space": "proj", "workers": [REVIEWER]}).json()
    assert p["state"] == "pending"
    sidecar = {"X-OpenWorker-Token": "sidecar-secret"}
    assert [x["id"] for x in client.get("/v1/team-proposals", headers=sidecar).json()["proposals"]] == [p["id"]]
    assert client.post(f"/v1/team-proposals/{p['id']}/decide", headers=sidecar, json={"approve": True}).status_code == 200
    mine = client.get(f"/v1/board/team-proposals/{p['id']}", headers=passport(key, "sb-lead"), params={"space": "proj"}).json()
    aid = mine["approvals"]["reviewer"]
    consume = {"approval_id": aid, "team": "proj", "lead": "lead", "worker": "reviewer", "digest": worker_digest(REVIEWER)}
    assert client.post("/v1/admin/approvals/consume", headers=ADMIN, json=consume).status_code == 200
    assert client.post("/v1/admin/approvals/consume", headers=ADMIN, json=consume).status_code == 403


def test_a_worker_cannot_propose_a_team(api):  # noqa: F811
    client, _, key = api
    r = client.post("/v1/board/team-proposals", headers=passport(key, "sb-rev"), json={"space": "proj", "workers": [REVIEWER]})
    assert r.status_code == 403


def test_the_user_decides_only_with_the_app_token(api):  # noqa: F811
    client, _, key = api
    p = client.post("/v1/board/team-proposals", headers=passport(key, "sb-lead"), json={"space": "proj", "workers": [REVIEWER]}).json()
    assert client.post(f"/v1/team-proposals/{p['id']}/decide", json={"approve": True}).status_code == 401
    r = client.post(f"/v1/team-proposals/{p['id']}/decide", headers=passport(key, "sb-lead"), json={"approve": True})
    assert r.status_code == 401
