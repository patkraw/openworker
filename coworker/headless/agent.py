"""`openworker agent`: one long-running team agent, for an OpenShell sandbox.

The agent never holds a board token: OpenShell's supervisor attaches a Sandbox Passport
to every board request outside the agent, and OpenShell's creation service registered
this sandbox with the board before starting it. The loop only makes outbound requests:

    pending work on the board  ->  one engine turn with the board tools  ->  consume

Model keys come from OpenShell providers (the process sees placeholders only)."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Optional


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openworker agent", description=__doc__.splitlines()[0])
    p.add_argument("--board-url", default=os.environ.get("OPENWORKER_BOARD_URL",
                                                         "http://host.openshell.internal:8765"))
    p.add_argument("--space", required=True, help="the team's board space")
    p.add_argument("--coworker", default="swe-worker", help="persona id")
    p.add_argument("--model", default=os.environ.get("OPENWORKER_MODEL"), help="e.g. anthropic:claude-sonnet-5-5")
    p.add_argument("--base-url", default=os.environ.get("OPENWORKER_BASE_URL"),
                   help="an OpenAI-compatible endpoint, e.g. https://inference-api.nvidia.com/v1;"
                        " the key comes from OPENAI_API_KEY (an OpenShell placeholder in a sandbox)")
    p.add_argument("--approval-mode", default="bypass-approvals",
                   choices=["auto-approve", "bypass-approvals"],
                   help="the sandbox is what contains the agent; nobody is present to approve")
    p.add_argument("--spawn-url", default=os.environ.get("OPENWORKER_SPAWN_URL", "http://host.openshell.internal:8766"),
                   help="OpenShell's creation service (Spawn Gate); used by a lead to staff workers")
    p.add_argument("--workspace", default=os.getcwd())
    p.add_argument("--poll-seconds", type=float, default=3.0)
    p.add_argument("--once", action="store_true", help="handle one batch of work, then exit")
    p.add_argument("--idle-exit-seconds", type=float, default=0, help="exit after this long with no work (0 = never)")
    p.add_argument("--scripted", default=None, help=argparse.SUPPRESS)  # tests: a fake model
    return p


def digest(events: list[dict[str, Any]], who: dict[str, Any], items: Optional[dict[int, dict]] = None) -> str:
    """The wake-up message. Each wake is a fresh conversation, so it carries the context
    from the board: the activity, and each item it touches (description, recent comments)."""
    lines = [f"You are {who.get('actor')} (role {who.get('role')}) on a team board. New activity on the board:"]
    for e in events:
        payload = e.get("payload") or {}
        detail = payload.get("body") or payload.get("comment") or payload.get("assignee") or ""
        lines.append(f"- {e.get('kind')} on item {e.get('item_id')} by {e.get('actor')}: {detail}".rstrip(": "))
    for item_id, item in sorted((items or {}).items()):
        lines.append(f"\nItem {item_id}: {item.get('title')} (state {item.get('state')}, assignee {item.get('assignee')})")
        if item.get("description"):
            lines.append(f"Description: {item['description']}")
        if item.get("criteria"):
            lines.append(f"Criteria: {item['criteria']}")
        for c in item.get("recent_comments", []):
            lines.append(f"  comment by {c.get('author')}: {c.get('body')}")
    lines.append("Decide what this activity asks of you and act on it now with your tools. "
                 "If an earlier instruction said to wait for this, the wait is over.")
    return "\n".join(lines)


def _latest_comments(dialect, space: str, item_id: int, keep: int = 10) -> list[dict]:
    """The last `keep` comments: pages are oldest-first, so read to the end."""
    tail: list[dict] = []
    after = 0
    for _ in range(100):   # at most 5000 comments
        page = dialect.comment_page(space, item_id, after_seq=after, limit=50)
        tail = (tail + page.get("comments", []))[-keep:]
        if not page.get("has_more"):
            break
        after = int(page.get("next_after_seq", after))
    return tail


def _item_context(dialect, space: str, events: list[dict[str, Any]]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for item_id in {e.get("item_id") for e in events if e.get("item_id")}:
        try:
            item = dict(dialect.get_item(space, int(item_id)))
            item["recent_comments"] = _latest_comments(dialect, space, int(item_id))
            out[int(item_id)] = item
        except Exception:  # context is a help, not a requirement
            continue
    return out


RETRY_NOTE = ("A previous attempt to handle this activity failed partway. Check the board "
              "for anything you already did before repeating an action.")
MAX_ATTEMPTS = 5


def make_provider(model: str, base_url: Optional[str] = None):
    """The model client. With a base URL, an OpenAI-compatible endpoint takes the bare
    model id (e.g. aws/anthropic/bedrock-claude-sonnet-5-5); otherwise OpenWorker's router."""
    from ..providers import ProviderRouter
    from ..providers.openai_provider import OpenAIProvider
    from ..secrets import SecretStore, state_dir
    from .runner import normalize_model

    if base_url:
        bare = model.split(":", 1)[1] if model.startswith("openai:") else model
        return OpenAIProvider(base_url=base_url), bare
    model = normalize_model(model)
    return (ProviderRouter(SecretStore(state_dir() / "secrets.json"),
                           default_provider=model.split(":", 1)[0] if ":" in model else "openai"), model)


def run(args: argparse.Namespace, *, client: Any = None) -> int:
    from ..agent import build_engine
    from ..agents.registry import get_agent
    from ..permissions import Mode
    from ..providers import ProviderRouter
    from ..secrets import SecretStore, state_dir
    from ..teams.dialect import RemoteDialect
    from ..teams.remote_tools import board_tools_over
    from ..teams.spawn_tools import spawn_tools
    from .runner import normalize_model, scripted_provider

    dialect = RemoteDialect(args.board_url, None, client=client)
    who = dialect.whoami()
    role = str(who.get("role", "worker"))
    print(f"openworker agent: {who.get('actor')} ({role}) on {args.space}", flush=True)

    if args.scripted:
        provider = scripted_provider(Path(args.scripted))
        model = normalize_model(args.model or "openai:gpt-test")
    else:
        if not args.model:
            print("openworker agent: --model is required", file=sys.stderr)
            return 2
        provider, model = make_provider(args.model, args.base_url)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    tools = board_tools_over(dialect, space=args.space, role=role) + (
        spawn_tools(args.spawn_url, team=args.space, board_url=args.board_url) if role == "lead" else [])

    def new_engine():
        # A fresh conversation per wake: the board is the agent's memory, turns stay small.
        engine = build_engine(
            agent=get_agent(args.coworker),
            workspace=workspace,
            model=model,
            mode=Mode(args.approval_mode),
            provider=provider,
            extra_tools=tools,
            session_id=f"agent-{uuid.uuid4().hex[:8]}",
        )
        engine.attendance = lambda: "auto"
        return engine

    idle_since = time.monotonic()
    attempts, consumed = 0, 0
    while True:
        page = dialect.pending_page(args.space)
        events, through = page["events"], page["through_seq"]
        if events:
            idle_since = time.monotonic()
            upto = max([through, *(int(e.get("seq", 0)) for e in events)])

            engine = new_engine()
            message = digest(events, who, _item_context(dialect, args.space, events))
            if attempts:
                message += "\n" + RETRY_NOTE
            failed: list[str] = []

            async def turn() -> None:
                async for ev in engine.run(message):
                    kind = getattr(ev.type, "value", str(ev.type))
                    data = ev.data if isinstance(ev.data, dict) else {}
                    if kind == "tool_proposed":
                        print(f"openworker agent: call {data.get('name') or data.get('tool')}", flush=True)
                    elif kind == "tool_finished":
                        result = str(data.get("result") or data.get("output") or data)[:300]
                        print(f"openworker agent: result {result}", flush=True)
                    elif kind in ("error", "assistant_message"):
                        print(f"openworker agent: {kind} {str(data.get('error') or data.get('text') or '')[:300]}", flush=True)
                    if kind == "error":
                        failed.append(str(data.get("error") or data))

            try:
                asyncio.run(turn())
            except Exception as error:  # a crashed turn is a failed turn
                failed.append(repr(error))
            if failed:
                # Keep the work: it is consumed only after a turn that finished cleanly.
                attempts += 1
                print(f"openworker agent: turn failed (attempt {attempts}): {failed[0][:200]}", flush=True)
                if args.once or attempts >= MAX_ATTEMPTS:
                    return 1
                time.sleep(min(60, 2 ** attempts))
                continue
            attempts = 0
            dialect.consume(args.space, upto)
            consumed = upto
            if args.once:
                return 0
        else:
            if through > consumed:
                # Nothing here for this agent: move past the quiet stretch.
                dialect.consume(args.space, through)
                consumed = through
            if args.idle_exit_seconds and time.monotonic() - idle_since > args.idle_exit_seconds:
                return 0
        time.sleep(args.poll_seconds)


def main(argv: Optional[list[str]] = None) -> None:
    raise SystemExit(run(_parser().parse_args(argv)))
