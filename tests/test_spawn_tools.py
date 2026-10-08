import pytest

from coworker.teams.spawn_tools import build_policy, spawn_tools

BOUNDARY = {
    "filesystem_policy": {"include_workdir": True, "read_only": ["/usr", "/etc"], "read_write": ["/tmp", "/dev/null", "/work"]},
    "network_policies": {
        "inference": {"endpoints": [{"host": "inference-api.nvidia.com", "port": 443}]},
        "github": {"endpoints": [{"host": "api.github.com", "port": 443}]},
    },
}


def test_policy_copies_only_the_chosen_boundary_pieces():
    p = build_policy(BOUNDARY, network=["inference"], writable=[])
    assert list(p["network_policies"]) == ["inference"]
    assert p["network_policies"]["inference"] == BOUNDARY["network_policies"]["inference"]
    assert p["filesystem_policy"]["read_write"] == ["/tmp", "/dev/null"]
    assert p["filesystem_policy"]["read_only"] == ["/usr", "/etc"]


def test_writable_path_must_be_writable_in_the_boundary():
    assert "/work" in build_policy(BOUNDARY, network=[], writable=["/work"])["filesystem_policy"]["read_write"]
    with pytest.raises(ValueError):
        build_policy(BOUNDARY, network=[], writable=["/etc"])


def test_unknown_network_entry_is_refused():
    with pytest.raises(ValueError):
        build_policy(BOUNDARY, network=["evil"], writable=[])


class FakeGate:
    def __init__(self):
        self.posted = []

    def get(self, path):
        return type("R", (), {"status_code": 200, "json": lambda self: BOUNDARY})()

    def post(self, path, json):
        self.posted.append((path, json))
        return type("R", (), {"status_code": 200, "json": lambda self: {"state": "running", "sandbox_id": "sb-1"}})()


def test_staff_worker_sends_a_proposed_policy_to_spawn_gate():
    gate = FakeGate()
    staff = {t.__name__: t for t in spawn_tools("http://gate", team="proj", client=gate)}["staff_worker"]
    result = staff("reviewer", "reviewer", "review item 1", network=["inference"])
    assert result == {"staffed": "reviewer", "state": "running", "sandbox_id": "sb-1"}
    path, body = gate.posted[0]
    assert path == "/v1/agents" and body["role"] == "worker" and body["persona"] == "reviewer"
    assert list(body["policy"]["network_policies"]) == ["inference"]


def test_staff_worker_reports_a_bad_request_instead_of_sending_it():
    gate = FakeGate()
    staff = {t.__name__: t for t in spawn_tools("http://gate", team="proj", client=gate)}["staff_worker"]
    assert "error" in staff("x", "reviewer", "t", network=["evil"])
    assert gate.posted == []


def test_proposals_name_the_teams_default_providers_so_the_user_approves_them():
    """Review finding 10: providers added after approval were never shown to the user."""
    class Gate(FakeGate):
        def get(self, path):
            body = {**BOUNDARY, "default_providers": ["openworker-nvidia"]}
            return type("R", (), {"status_code": 200, "json": lambda self: body})()

    class Board:
        def post(self, path, json):
            self.sent = json
            return type("R", (), {"status_code": 200, "json": lambda self: {"id": "tp_1", "state": "pending"}})()

    board = Board()
    propose = {t.__name__: t for t in spawn_tools("http://gate", team="proj", client=Gate(),
                                                   board_client=board)}["propose_sandbox_team"]
    propose([{"name": "reviewer", "persona": "reviewer", "network": ["inference"]},
             {"name": "patcher", "persona": "swe-worker", "providers": ["github"]}])
    by_name = {w["name"]: w for w in board.sent["workers"]}
    assert by_name["reviewer"]["providers"] == ["openworker-nvidia"]
    assert by_name["patcher"]["providers"] == ["github"]
