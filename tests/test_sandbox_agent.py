"""`openworker agent` against the real board API with a Sandbox Passport and a scripted
model: it wakes on its assignment, comments through the board tools, and consumes."""

import json

import pytest

from coworker.teams.model import Actor, Role
from test_board_passport import ADMIN, AUD, passport  # noqa: F401  (fixtures and helpers)
from test_board_passport import api  # noqa: F401

USER = Actor("user", Role.USER)


class PassportClient:
    """Wraps the test client so every request carries this sandbox's Passport,
    as OpenShell's supervisor does outside the agent."""

    def __init__(self, client, key, sandbox_id):
        self._client, self._key, self._sid = client, key, sandbox_id
        self.headers = {}

    def get(self, path, params=None):
        return self._client.get(path, params=params, headers=passport(self._key, self._sid))

    def post(self, path, json=None):
        return self._client.post(path, json=json, headers=passport(self._key, self._sid))


def test_agent_wakes_on_its_assignment_and_comments_as_itself(api, tmp_path):  # noqa: F811
    client, manager, key = api
    item = manager.team_store.create_item("proj", USER, title="Fix the bug", criteria="tests pass")
    manager.team_store.assign("proj", USER, item["id"], "reviewer")
    script = tmp_path / "turns.json"
    script.write_text(json.dumps([
        {"tool_calls": [{"name": "board_comment", "arguments": {"id": item["id"], "body": "Reviewed: LGTM"}}]},
        {"text": "Done."},
    ]))
    from coworker.headless.agent import _parser, run

    args = _parser().parse_args(["--space", "proj", "--board-url", "http://board.test", "--once",
                                 "--scripted", str(script), "--workspace", str(tmp_path / "ws"),
                                 "--poll-seconds", "0"])
    assert run(args, client=PassportClient(client, key, "sb-rev")) == 0

    comments = manager.team_store.comment_page("proj", item["id"], actor=USER)["comments"]
    assert [(c["author"], c["body"]) for c in comments] == [("reviewer", "Reviewed: LGTM")]
    assert manager.team_store.feed_for("proj", "reviewer") == []  # consumed


def test_worker_is_not_offered_assign(api):  # noqa: F811
    from coworker.teams.remote_tools import board_tools_over

    names = {t.__name__ for t in board_tools_over(object(), space="proj", role="worker")}
    assert "board_assign" not in names and "board_comment" in names


def test_a_base_url_sends_the_bare_model_id_to_an_openai_compatible_endpoint():
    from coworker.headless.agent import make_provider

    provider, model = make_provider("openai:aws/anthropic/bedrock-claude-sonnet-5-5",
                                    "https://inference-api.nvidia.com/v1")
    assert model == "aws/anthropic/bedrock-claude-sonnet-5-5"
    assert provider._base_url == "https://inference-api.nvidia.com/v1"


def test_a_second_wake_runs_a_second_turn(api, tmp_path):  # noqa: F811
    client, manager, key = api
    item = manager.team_store.create_item("proj", USER, title="First", criteria="x")
    manager.team_store.assign("proj", USER, item["id"], "reviewer")
    script = tmp_path / "turns.json"
    script.write_text(json.dumps([
        {"tool_calls": [{"name": "board_comment", "arguments": {"id": item["id"], "body": "first turn"}}]},
        {"text": "Waiting."},
        {"tool_calls": [{"name": "board_comment", "arguments": {"id": item["id"], "body": "second turn"}}]},
        {"text": "Done."},
    ]))
    from coworker.headless import agent as agent_mod

    calls = {"n": 0}
    real_sleep = agent_mod.time.sleep

    def sleep_and_poke(seconds):
        calls["n"] += 1
        if calls["n"] == 1:   # after the first turn, the user comments: a second wake
            manager.team_store.comment("proj", USER, item["id"], "approved, go ahead")
        real_sleep(0)

    agent_mod.time.sleep = sleep_and_poke
    try:
        args = agent_mod._parser().parse_args([
            "--space", "proj", "--board-url", "http://board.test", "--scripted", str(script),
            "--workspace", str(tmp_path / "ws"), "--poll-seconds", "0", "--idle-exit-seconds", "0.5"])
        agent_mod.run(args, client=PassportClient(client, key, "sb-rev"))
    finally:
        agent_mod.time.sleep = real_sleep
    bodies = [c["body"] for c in manager.team_store.comment_page("proj", item["id"], actor=USER)["comments"]]
    assert bodies == ["first turn", "approved, go ahead", "second turn"]
