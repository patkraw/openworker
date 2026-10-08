"""OpenShell Sandbox Passport: board access for agents running in OpenShell sandboxes.

When OpenWorker's agents each run in their own OpenShell sandbox, the board does not
hand them bearer tokens. Instead OpenShell's supervisor attaches a signed caller identity
(`X-OpenShell-Caller`) outside the agent, naming the calling sandbox. OpenShell's creation
service (Spawn Gate) registers each sandbox here with its team, agent name and role, and
removes it when the agent stops. A Passport caller is locked to its team's space.

Configured by environment (all three, or the feature is off):
  OPENWORKER_PASSPORT_PUBLIC_KEY  PEM file of the Passport signer's Ed25519 public key
  OPENWORKER_PASSPORT_AUDIENCE    this board as callers address it, e.g. host.openshell.internal:8765
  OPENWORKER_BOARD_ADMIN_TOKEN    shared with Spawn Gate for registrations
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization

from .model import Actor, Role

HEADER = "X-OpenShell-Caller"
ADMIN_HEADER = "X-Board-Admin"
ISSUER = "openshell-teams/passport"
LEEWAY_SECONDS = 5


class PassportError(Exception):
    pass


@dataclass(frozen=True)
class SandboxAgent:
    sandbox_id: str
    team: str
    space: str
    name: str
    role: str

    def actor(self) -> Actor:
        return Actor(self.name, Role(self.role))


def _unb64(part: str) -> bytes:
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


class PassportVerifier:
    def __init__(self, public_key_pem: bytes, audience: str):
        self._key = serialization.load_pem_public_key(public_key_pem)
        self.audience = audience

    def verify(self, token: str) -> dict:
        try:
            header_b64, claims_b64, sig_b64 = token.split(".")
            header = json.loads(_unb64(header_b64))
            claims = json.loads(_unb64(claims_b64))
        except (ValueError, json.JSONDecodeError) as error:
            raise PassportError("malformed Passport") from error
        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise PassportError("malformed Passport")
        if header.get("alg") != "EdDSA":
            raise PassportError("unexpected algorithm")
        try:
            self._key.verify(_unb64(sig_b64), f"{header_b64}.{claims_b64}".encode())
        except InvalidSignature as error:
            raise PassportError("bad signature") from error
        now = time.time()
        if claims.get("iss") != ISSUER:
            raise PassportError("wrong issuer")
        if claims.get("aud") != self.audience:
            raise PassportError("Passport is for another service")
        if not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < now - LEEWAY_SECONDS:
            raise PassportError("expired")
        if not isinstance(claims.get("sbx"), str) or not claims["sbx"]:
            raise PassportError("no sandbox id")
        return claims


class SandboxAgents:
    """Registrations from Spawn Gate: sandbox id -> agent. Persisted in the state dir."""

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._agents: dict[str, SandboxAgent] = {}
        if path.exists():
            for row in json.loads(path.read_text() or "[]"):
                self._agents[row["sandbox_id"]] = SandboxAgent(**row)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([a.__dict__ for a in self._agents.values()]))
        os.chmod(tmp, 0o600)
        tmp.replace(self._path)

    def register(self, agent: SandboxAgent) -> None:
        with self._lock:
            self._agents[agent.sandbox_id] = agent
            self._save()

    def remove(self, sandbox_id: str) -> None:
        with self._lock:
            self._agents.pop(sandbox_id, None)
            self._save()

    def get(self, sandbox_id: str) -> Optional[SandboxAgent]:
        with self._lock:
            return self._agents.get(sandbox_id)


@dataclass
class PassportAuth:
    verifier: PassportVerifier
    agents: SandboxAgents
    admin_token: str

    @classmethod
    def from_env(cls, state_dir: Path) -> Optional["PassportAuth"]:
        key = os.environ.get("OPENWORKER_PASSPORT_PUBLIC_KEY")
        audience = os.environ.get("OPENWORKER_PASSPORT_AUDIENCE")
        admin = os.environ.get("OPENWORKER_BOARD_ADMIN_TOKEN")
        if not (key and audience and admin):
            return None
        return cls(PassportVerifier(Path(key).read_bytes(), audience),
                   SandboxAgents(state_dir / "sandbox-agents.json"), admin)

    def is_admin(self, token: str) -> bool:
        return bool(token) and secrets.compare_digest(token, self.admin_token)

    def caller(self, token: str) -> SandboxAgent:
        """Raises PassportError (invalid -> 401) or LookupError (not registered -> 403)."""
        claims = self.verifier.verify(token)
        agent = self.agents.get(claims["sbx"])
        if agent is None:
            raise LookupError("sandbox is not registered with this board")
        return agent
