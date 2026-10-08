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
    assert "board_link" not in names and "board_create" not in names


def test_lead_links_a_review_item_to_the_work_it_reviews():
    from coworker.teams.remote_tools import board_tools_over

    class Dialect:
        def link(self, space, src, kind, dst):
            self.call = (space, src, kind, dst)
            return {"seq": 7, "kind": "item_linked", "item_id": src}

    d = Dialect()
    link = {t.__name__: t for t in board_tools_over(d, space="proj", role="lead")}["board_link"]
    link(2, 1)
    assert d.call == ("proj", 2, "parent", 1)


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


# --- review findings ------------------------------------------------------------------------

def _args(tmp_path, script, *extra):
    from coworker.headless.agent import _parser
    return _parser().parse_args(["--space", "proj", "--board-url", "http://board.test",
                                 "--scripted", str(script), "--workspace", str(tmp_path / "ws"),
                                 "--poll-seconds", "0", *extra])


def test_a_failed_turn_keeps_the_work_and_reports_failure(api, tmp_path):  # noqa: F811
    """Review finding 8: an error event still consumed the batch and exited 0."""
    client, manager, key = api
    item = manager.team_store.create_item("proj", USER, title="Fix", criteria="x")
    manager.team_store.assign("proj", USER, item["id"], "reviewer")
    script = tmp_path / "turns.json"
    script.write_text(json.dumps([{"error": "provider unavailable"}]))   # the model call fails
    from coworker.headless.agent import run
    assert run(_args(tmp_path, script, "--once"), client=PassportClient(client, key, "sb-rev")) != 0
    assert manager.team_store.feed_for("proj", "reviewer") != []   # still pending


def test_a_page_of_unrelated_events_does_not_hide_later_work(api, tmp_path):  # noqa: F811
    """Review finding 16: the feed limited raw events before filtering, so a quiet first
    page returned nothing and the cursor never moved."""
    client, manager, key = api
    for i in range(205):
        manager.team_store.create_item("proj", USER, title=f"noise {i}", criteria="x")
    item = manager.team_store.create_item("proj", USER, title="Real work", criteria="x")
    manager.team_store.assign("proj", USER, item["id"], "reviewer")
    script = tmp_path / "turns.json"
    script.write_text(json.dumps([
        {"tool_calls": [{"name": "board_comment", "arguments": {"id": item["id"], "body": "on it"}}]},
        {"text": "Done."},
    ]))
    from coworker.headless.agent import run
    run(_args(tmp_path, script, "--idle-exit-seconds", "1"), client=PassportClient(client, key, "sb-rev"))
    bodies = [c["body"] for c in manager.team_store.comment_page("proj", item["id"], actor=USER)["comments"]]
    assert bodies == ["on it"]


def test_the_wake_up_context_has_the_latest_comments(api):  # noqa: F811
    """Review finding 17: 'recent comments' were the first page, i.e. the oldest."""
    client, manager, key = api
    item = manager.team_store.create_item("proj", USER, title="Long thread", criteria="x")
    manager.team_store.assign("proj", USER, item["id"], "reviewer")
    for i in range(60):
        manager.team_store.comment("proj", USER, item["id"], f"comment {i}")
    from coworker.headless.agent import _item_context
    from coworker.teams.dialect import RemoteDialect
    dialect = RemoteDialect("http://board.test", None, client=PassportClient(client, key, "sb-rev"))
    ctx = _item_context(dialect, "proj", [{"item_id": item["id"]}])
    bodies = [c["body"] for c in ctx[item["id"]]["recent_comments"]]
    assert bodies[-1] == "comment 59" and len(bodies) == 10
