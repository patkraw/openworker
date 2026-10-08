"""Lead tools for staffing a team through OpenShell's creation service (Spawn Gate).

The lead proposes each worker's sandbox policy, built only from pieces of the team
boundary (so OpenShell's prover can check it), and Spawn Gate admits the worker only if
the policy is proven inside the boundary. Identity is the Sandbox Passport OpenShell
attaches to every request outside the agent; this client sends no credentials."""

from __future__ import annotations

import uuid
from typing import Any, Optional

ALWAYS_WRITABLE = ("/tmp", "/dev/null")


def build_policy(boundary: dict, *, network: list[str], writable: list[str]) -> dict:
    """A worker policy made of the boundary's own pieces: its read-only paths, the chosen
    writable paths, and the chosen network entries, copied exactly."""
    fs = boundary.get("filesystem_policy", {})
    entries = boundary.get("network_policies", {})
    unknown = sorted(set(network) - set(entries))
    if unknown:
        raise ValueError(f"not in the team boundary: {unknown}; choose from {sorted(entries)}")
    bad_paths = sorted(set(writable) - set(fs.get("read_write", [])))
    if bad_paths:
        raise ValueError(f"not writable in the team boundary: {bad_paths}")
    policy: dict[str, Any] = {
        "version": 1,
        "filesystem_policy": {
            "include_workdir": fs.get("include_workdir", True),
            "read_only": list(fs.get("read_only", [])),
            "read_write": [p for p in fs.get("read_write", []) if p in ALWAYS_WRITABLE or p in writable],
        },
    }
    if network:
        policy["network_policies"] = {name: entries[name] for name in network}
    return policy


def spawn_tools(spawn_url: str, *, team: str, client: Optional[Any] = None,
                board_url: Optional[str] = None, board_client: Optional[Any] = None) -> list:
    import httpx

    http = client or httpx.Client(base_url=spawn_url.rstrip("/"), timeout=900)
    board = board_client or (httpx.Client(base_url=board_url.rstrip("/"), timeout=60) if board_url else None)

    def team_boundary() -> Any:
        """Show the most any worker on this team may have: filesystem paths, the named
        network entries, providers and roles. Workers are built from these pieces."""
        r = http.get(f"/v1/teams/{team}/boundary")
        return r.json() if r.status_code == 200 else {"error": r.text, "status": r.status_code}

    def staff_worker(name: str, persona: str, task: str, network: Optional[list[str]] = None,
                     writable: Optional[list[str]] = None, providers: Optional[list[str]] = None) -> Any:
        """Create a worker in its own sandbox. `network` names entries from team_boundary
        that the worker needs (leave out what it does not need); `writable` names writable
        paths it needs; `persona` is its OpenWorker persona (e.g. reviewer, swe-worker).
        OpenShell proves the policy is inside the team boundary before the worker starts;
        then create a board item and assign it to the worker."""
        boundary = team_boundary()
        if "error" in boundary:
            return boundary
        try:
            policy = build_policy(boundary, network=list(network or []), writable=list(writable or []))
        except ValueError as error:
            return {"error": str(error)}
        body = {"team": team, "name": name, "role": "worker", "persona": persona, "task": task,
                "policy": policy, "providers": list(providers or []),
                "request_id": f"{name}-{uuid.uuid4().hex[:8]}"}
        r = http.post("/v1/agents", json=body)
        if r.status_code == 200:
            return {"staffed": name, **{k: v for k, v in r.json().items() if k in ("state", "sandbox_id")}}
        return {"error": "Spawn Gate refused", "status": r.status_code, "detail": r.json().get("detail", r.text)}

    def propose_sandbox_team(workers: list[dict]) -> Any:
        """Propose your team for the user to approve. Each worker is a dict with `name`,
        `persona`, `task`, and optionally `network` (entry names from team_boundary) and
        `writable` (paths). Nothing is created until the user approves; then call
        staff_approved_team with the returned proposal id. (Named apart from OpenWorker's in-app propose_team.)"""
        if board is None:
            return {"error": "no board address configured"}
        boundary = team_boundary()
        if "error" in boundary:
            return boundary
        proposed = []
        for w in workers:
            try:
                policy = build_policy(boundary, network=list(w.get("network") or []),
                                      writable=list(w.get("writable") or []))
            except ValueError as error:
                return {"error": f"{w.get('name')}: {error}"}
            proposed.append({"name": w["name"], "persona": w.get("persona", ""), "role": "worker",
                             "task": w.get("task", ""), "policy": policy, "providers": list(w.get("providers") or [])})
        r = board.post("/v1/board/team-proposals", json={"space": team, "workers": proposed})
        data = r.json()
        if r.status_code != 200:
            return {"error": data.get("error", r.text)}
        return {"proposal_id": data["id"], "state": data["state"],
                "next": "wait for the user to approve, then call staff_approved_team"}

    def staff_approved_team(proposal_id: str) -> Any:
        """Create the workers of a team proposal the user approved, each with its approval."""
        if board is None:
            return {"error": "no board address configured"}
        r = board.get(f"/v1/board/team-proposals/{proposal_id}", params={"space": team})
        proposal = r.json()
        if r.status_code != 200:
            return {"error": proposal.get("error", r.text)}
        if proposal.get("state") != "approved":
            return {"state": proposal.get("state"), "note": "not approved yet; nothing was created"}
        results = {}
        for w in proposal["workers"]:
            body = {"team": team, "name": w["name"], "role": w.get("role", "worker"), "persona": w.get("persona", ""),
                    "task": w.get("task", ""), "policy": w["policy"], "providers": w.get("providers", []),
                    "approval_id": proposal["approvals"][w["name"]],
                    "request_id": f"{proposal_id}-{w['name']}"}
            rr = http.post("/v1/agents", json=body)
            results[w["name"]] = (rr.json().get("state") if rr.status_code == 200
                                  else {"refused": rr.json().get("detail", rr.text)})
        return {"staffed": results, "next": "create board items and assign them to the workers"}

    return [team_boundary, staff_worker, propose_sandbox_team, staff_approved_team]
