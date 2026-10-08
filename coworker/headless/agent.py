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


def digest(events: list[dict[str, Any]], who: dict[str, Any]) -> str:
    lines = [f"You are {who.get('actor')} (role {who.get('role')}) on a team board. New activity:"]
    for e in events:
        payload = e.get("payload") or {}
        detail = payload.get("body") or payload.get("comment") or payload.get("assignee") or ""
        lines.append(f"- {e.get('kind')} on item {e.get('item_id')} by {e.get('actor')}: {detail}".rstrip(": "))
    lines.append("Use the board tools to act on it, then stop.")
    return "\n".join(lines)


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
    engine = build_engine(
        agent=get_agent(args.coworker),
        workspace=workspace,
        model=model,
        mode=Mode(args.approval_mode),
        provider=provider,
        extra_tools=board_tools_over(dialect, space=args.space, role=role) + (
            spawn_tools(args.spawn_url, team=args.space) if role == "lead" else []),
        session_id=f"agent-{uuid.uuid4().hex[:8]}",
    )
    engine.attendance = lambda: "auto"

    idle_since = time.monotonic()
    while True:
        events = dialect.pending(args.space)
        if events:
            idle_since = time.monotonic()
            upto = max(int(e.get("seq", 0)) for e in events)

            async def turn() -> None:
                async for ev in engine.run(digest(events, who)):
                    kind = getattr(ev.type, "value", str(ev.type))
                    if kind in ("tool_proposed", "error", "message_end"):
                        print(f"openworker agent: {kind}", flush=True)

            asyncio.run(turn())
            dialect.consume(args.space, upto)
            if args.once:
                return 0
        elif args.idle_exit_seconds and time.monotonic() - idle_since > args.idle_exit_seconds:
            return 0
        time.sleep(args.poll_seconds)


def main(argv: Optional[list[str]] = None) -> None:
    raise SystemExit(run(_parser().parse_args(argv)))
