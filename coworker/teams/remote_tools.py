"""Board tools for an agent that reaches the board over HTTP (a `RemoteDialect`),
pre-bound to its space. Used by `openworker agent` inside an OpenShell sandbox, where
identity comes from the Sandbox Passport OpenShell attaches outside the agent.

The board re-checks every call; this set is role-filtered for convenience only."""

from __future__ import annotations

from typing import Any, Optional

from .model import BoardError
from .tools import item_snapshot, mutation_receipt


def _safe(func, *args, **kwargs) -> Any:
    try:
        return func(*args, **kwargs)
    except (BoardError, ValueError) as error:
        return {"error": str(error)}


def board_tools_over(dialect, *, space: str, role: str) -> list:
    def board_list(state: str = "", assignee: str = "") -> Any:
        """List work items on the team board, optionally filtered by state
        (open/in_progress/blocked/review/done/canceled) or assignee."""
        items = _safe(dialect.list_items, space, state=state or None, assignee=assignee or None)
        if not isinstance(items, list):
            return items
        return {"items": [item_snapshot(i, brief=True) for i in items]}

    def board_get(id: int) -> Any:
        """Read one work item: title, acceptance criteria, state, assignee and links."""
        return _safe(dialect.get_item, space, int(id))

    def board_comments(id: int) -> Any:
        """Read the comments on a work item, oldest first."""
        return _safe(dialect.comment_page, space, int(id))

    def board_comment(id: int, body: str) -> Any:
        """Add a comment to a work item. You are always the author."""
        result = _safe(dialect.comment, space, int(id), body)
        return result if "error" in result else mutation_receipt(result)

    def board_move(id: int, state: str, comment: str = "") -> Any:
        """Move a work item to another state (workers: in_progress, blocked or review).
        Add a comment saying why."""
        result = _safe(dialect.transition, space, int(id), state, comment=comment or None)
        return result if "error" in result else mutation_receipt(result)

    def board_create(title: str, criteria: str, description: str = "", parent: Optional[int] = None) -> Any:
        """Create a work item. `criteria` is what gets verified before it can be done."""
        result = _safe(dialect.create_item, space, title=title, criteria=criteria,
                       description=description, parent=parent)
        return result if "error" in result else mutation_receipt(result)

    def board_assign(id: int, assignee: str) -> Any:
        """Assign a work item to a team member (lead only)."""
        result = _safe(dialect.assign, space, int(id), assignee)
        return result if "error" in result else mutation_receipt(result)

    def board_link(id: int, parent_id: int) -> Any:
        """Link item `id` under item `parent_id` (lead only). Whoever is assigned `id`
        can then read and comment on `parent_id`, e.g. a reviewer on the work it reviews."""
        result = _safe(dialect.link, space, int(id), "parent", int(parent_id))
        return result if "error" in result else mutation_receipt(result)

    tools = [board_list, board_get, board_comments, board_comment, board_move]
    if role == "lead":
        # Workers are not offered what their role cannot do (Channel Guard refuses it anyway).
        tools += [board_create, board_assign, board_link]
    return tools
