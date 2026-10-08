"""Team proposals and approval IDs for leads in OpenShell sandboxes.

A sandboxed lead proposes its team (each worker's name, persona, role, sandbox policy and
providers). The user approves the proposal on the approval card. The server then issues
one single-use approval ID per worker, bound to the lead, the team and a digest of exactly
what the user saw. OpenShell's creation service (Spawn Gate) consumes the ID before it
creates the worker; a changed policy, a reused ID or another lead's ID is refused."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

APPROVAL_LIFETIME_SECONDS = 15 * 60


class ApprovalError(Exception):
    pass


def worker_digest(worker: dict[str, Any]) -> str:
    """What the user approved for one worker. Spawn Gate computes the same digest."""
    canonical = {"name": worker["name"], "role": worker.get("role", "worker"),
                 "persona": worker.get("persona", ""), "policy": worker["policy"],
                 "providers": sorted(worker.get("providers") or [])}
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class ApprovalStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS proposals (
                id TEXT PRIMARY KEY, space TEXT NOT NULL, lead TEXT NOT NULL,
                workers TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL, decided_by TEXT);
            CREATE TABLE IF NOT EXISTS approvals (
                id TEXT PRIMARY KEY, proposal TEXT NOT NULL, space TEXT NOT NULL, lead TEXT NOT NULL,
                worker TEXT NOT NULL, digest TEXT NOT NULL, expires REAL NOT NULL, used REAL);
        """)
        # The proposing lead's sandbox: a replacement lead with the same name is another lead.
        for table in ("proposals", "approvals"):
            cols = {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}
            if "lead_instance" not in cols:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN lead_instance TEXT")

    def propose(self, space: str, lead: str, workers: list[dict[str, Any]],
                lead_instance: Optional[str] = None) -> dict[str, Any]:
        if not workers:
            raise ApprovalError("a team proposal needs at least one worker")
        names = [w.get("name") for w in workers]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ApprovalError("every worker needs a unique name")
        for w in workers:
            if not isinstance(w.get("policy"), dict):
                raise ApprovalError(f"worker {w.get('name')} needs a policy")
        pid = "tp_" + secrets.token_hex(8)
        with self._lock:
            self._db.execute("INSERT INTO proposals (id, space, lead, workers, state, created, decided_by,"
                             " lead_instance) VALUES (?, ?, ?, ?, 'pending', ?, NULL, ?)",
                             (pid, space, lead, json.dumps(workers), time.time(), lead_instance))
        return self.get(pid)

    def get(self, pid: str) -> Optional[dict[str, Any]]:
        row = self._db.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
        if not row:
            return None
        out = {"id": row["id"], "space": row["space"], "lead": row["lead"], "state": row["state"],
               "workers": json.loads(row["workers"]), "decided_by": row["decided_by"]}
        if row["state"] == "approved":
            out["approvals"] = {r["worker"]: r["id"] for r in self._db.execute(
                "SELECT id, worker FROM approvals WHERE proposal=?", (pid,))}
        return out

    def pending(self) -> list[dict[str, Any]]:
        return [self.get(r["id"]) for r in self._db.execute(
            "SELECT id FROM proposals WHERE state='pending' ORDER BY created")]

    def decide(self, pid: str, *, approve: bool, by: str) -> dict[str, Any]:
        """Record the user's decision and, when approved, one approval ID per worker, in one
        transaction: nobody can see 'approved' without its IDs."""
        with self._lock:
            row = self._db.execute("SELECT * FROM proposals WHERE id=?", (pid,)).fetchone()
            if not row:
                raise ApprovalError("no such proposal")
            if row["state"] != "pending":
                raise ApprovalError(f"proposal is already {row['state']}")
            # Everything that can fail happens before anything is written.
            ids = [("ap_" + secrets.token_urlsafe(16), w["name"], worker_digest(w))
                   for w in json.loads(row["workers"])] if approve else []
            expires = time.time() + APPROVAL_LIFETIME_SECONDS
            self._db.execute("BEGIN IMMEDIATE")
            try:
                cur = self._db.execute("UPDATE proposals SET state=?, decided_by=? WHERE id=? AND state='pending'",
                                       ("approved" if approve else "rejected", by, pid))
                if cur.rowcount != 1:
                    raise ApprovalError("proposal was decided concurrently")
                for aid, worker, digest in ids:
                    self._db.execute("INSERT INTO approvals (id, proposal, space, lead, worker, digest, expires,"
                                     " used, lead_instance) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)",
                                     (aid, pid, row["space"], row["lead"], worker, digest, expires,
                                      row["lead_instance"]))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
        return self.get(pid)

    def consume(self, approval_id: str, *, space: str, lead: str, worker: str, digest: str,
                lead_instance: Optional[str] = None) -> None:
        """Single use. Raises ApprovalError unless everything matches what the user approved."""
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT * FROM approvals WHERE id=?", (approval_id,)).fetchone()
                problem = (
                    "unknown approval" if not row else
                    "approval already used" if row["used"] else
                    "approval expired" if row["expires"] < time.time() else
                    "approval is for another team" if row["space"] != space else
                    "approval is for another lead" if row["lead"] != lead else
                    "approval is for another lead instance" if row["lead_instance"] and
                    row["lead_instance"] != lead_instance else
                    "approval is for another worker" if row["worker"] != worker else
                    "this worker differs from what the user approved" if row["digest"] != digest else None)
                if problem:
                    raise ApprovalError(problem)
                self._db.execute("UPDATE approvals SET used=? WHERE id=?", (time.time(), approval_id))
                self._db.execute("COMMIT")
            except Exception:
                self._db.execute("ROLLBACK")
                raise
