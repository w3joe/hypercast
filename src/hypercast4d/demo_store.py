"""Durable demo sessions and quotas. One API container owns this SQLite store."""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path

from .demo_policy import LIMITS


class LimitError(ValueError):
    pass


class DemoStore:
    def __init__(self, root: Path, pepper: str, *, persist=lambda: None, clock=time.time):
        if len(pepper) < 32:
            raise ValueError("DEMO_SESSION_SECRET must contain at least 32 random characters.")
        self.root, self.pepper, self.persist, self.clock = root, pepper, persist, clock
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        with self.transaction() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS challenges (
                    id TEXT PRIMARY KEY, owner TEXT, digest TEXT, expires REAL, attempts INTEGER);
                CREATE TABLE IF NOT EXISTS sessions (digest TEXT PRIMARY KEY, owner TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS events (kind TEXT, owner TEXT, created REAL);
                CREATE INDEX IF NOT EXISTS event_lookup ON events(kind, owner, created);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, owner TEXT, created REAL, call_id TEXT, record TEXT);
                CREATE TABLE IF NOT EXISTS architectures (
                    id TEXT PRIMARY KEY, owner TEXT, updated REAL, record TEXT);
            """)

    @contextmanager
    def transaction(self):
        # Modal API is deployed with max_containers=1 and strategy=recreate.
        # A process lock serializes requests; never mount this DB in GPU workers.
        with self.lock:
            db = sqlite3.connect(self.root / "demo.sqlite", timeout=10)
            db.row_factory = sqlite3.Row
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                changed = db.total_changes
                db.commit()
            except BaseException:
                db.rollback()
                raise
            finally:
                db.close()
            if changed:
                self.persist()

    def digest(self, text: str) -> str:
        return hmac.new(self.pepper.encode(), text.encode(), hashlib.sha256).hexdigest()

    def _count(self, db, kind, owner, since):
        sql = "SELECT COUNT(*) FROM events WHERE kind=? AND created>=?"
        args = [kind, since]
        if owner is not None:
            sql += " AND owner=?"
            args.append(owner)
        return db.execute(sql, args).fetchone()[0]

    def _reserve(self, db, kind, owner, rules):
        now = self.clock()
        for scope, since, limit, message in rules:
            if self._count(db, kind, scope, since) >= limit:
                raise LimitError(message)
        db.execute("INSERT INTO events VALUES (?, ?, ?)", (kind, owner, now))

    def period(self, monthly=False):
        now = datetime.fromtimestamp(self.clock(), UTC)
        return now.replace(day=1 if monthly else now.day, hour=0, minute=0, second=0, microsecond=0).timestamp()

    def challenge(self, email: str) -> tuple[str, str, str]:
        email = email.strip().lower()
        if len(email) > 254 or not re.fullmatch(r"[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+", email):
            raise ValueError("Enter a valid email address.")
        owner = self.digest("email:" + email)
        challenge_id = secrets.token_urlsafe(32)
        code = f"{secrets.randbelow(1_000_000):06d}"
        with self.transaction() as db:
            self._reserve(db, "email", owner, [
                (owner, self.clock() - 3600, 3, "Please wait before requesting another email code."),
                (None, self.clock() - 3600, 30, "Email verification is busy. Please try again later."),
                (None, self.period(), 100, "Today's demo email limit has been reached."),
            ])
            db.execute("DELETE FROM challenges WHERE owner=? OR expires<?", (owner, self.clock()))
            db.execute("INSERT INTO challenges VALUES (?, ?, ?, ?, 0)",
                       (challenge_id, owner, self.digest(challenge_id + ":" + code), self.clock() + 600))
        return challenge_id, code, email

    def verify(self, challenge_id: str, code: str) -> str:
        token = None
        with self.transaction() as db:
            row = db.execute("SELECT * FROM challenges WHERE id=?", (challenge_id,)).fetchone()
            if row and row["expires"] > self.clock() and row["attempts"] < 5:
                db.execute("UPDATE challenges SET attempts=attempts+1 WHERE id=?", (challenge_id,))
                if hmac.compare_digest(row["digest"], self.digest(challenge_id + ":" + code)):
                    token = secrets.token_urlsafe(32)
                    db.execute("DELETE FROM challenges WHERE id=?", (challenge_id,))
                    db.execute("DELETE FROM sessions WHERE expires<?", (self.clock(),))
                    db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                               (self.digest(token), row["owner"], self.clock() + 7 * 86400))
        # Failed attempts must commit too; raising inside the transaction rolls them back.
        if token is None:
            raise ValueError("That code is invalid or expired. Request a new code after five attempts.")
        return token

    def owner(self, token: str | None) -> str | None:
        if not token or len(token) > 128:
            return None
        with self.transaction() as db:
            row = db.execute("SELECT owner FROM sessions WHERE digest=? AND expires>?",
                             (self.digest(token), self.clock())).fetchone()
            return row[0] if row else None

    def logout(self, token: str):
        with self.transaction() as db:
            db.execute("DELETE FROM sessions WHERE digest=?", (self.digest(token),))

    def quota(self, owner: str):
        with self.transaction() as db:
            used = self._count(db, "run", owner, self.period())
            monthly = self._count(db, "run", None, self.period(True))
        return {"remaining_runs": max(0, LIMITS["runs_per_day"] - used),
                "demo_available": monthly < LIMITS["runs_per_month_total"]}

    def reserve_inspection(self, owner):
        with self.transaction() as db:
            self._reserve(db, "inspect", owner, [
                (owner, self.clock() - 60, 20, "Please wait a minute before checking more architectures."),
                (owner, self.period(), 100, "Today's architecture-check limit has been reached."),
                (None, self.period(True), 2000, "The demo architecture-check allowance is exhausted this month."),
            ])

    def reserve_run(self, owner: str, record: dict):
        """Reserve before dispatch; failed/cancelled runs also consume the allowance."""
        with self.transaction() as db:
            active = db.execute("SELECT owner FROM jobs WHERE json_extract(record, '$.status.state') IN ('queued','running','starting')").fetchall()
            if any(row[0] == owner for row in active):
                raise LimitError("You already have an experiment queued or running.")
            if len(active) >= LIMITS["queue_size"]:
                raise LimitError("The demo queue is full. Please try again later.")
            self._reserve(db, "run", owner, [
                (owner, self.period(), LIMITS["runs_per_day"], "Your three demo runs for today have been used. Try again tomorrow (UTC)."),
                (None, self.period(True), LIMITS["runs_per_month_total"], "The shared demo allowance has been used for this month."),
            ])
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, NULL, ?)",
                       (record["id"], owner, self.clock(), json.dumps(record, allow_nan=False)))

    def jobs(self, owner: str | None, *, active_only=False):
        with self.transaction() as db:
            rows = db.execute("SELECT * FROM jobs" + (" WHERE owner=?" if owner else "") + " ORDER BY created DESC",
                              (owner,) if owner else ()).fetchall()
        result = [dict(row) for row in rows]
        for row in result:
            row["record"] = json.loads(row["record"])
        return [r for r in result if not active_only or r["record"]["status"]["state"] in {"queued", "running", "starting"}]

    def job(self, owner, job_id):
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND owner=?", (job_id, owner)).fetchone()
        if not row:
            raise KeyError(job_id)
        return {**dict(row), "record": json.loads(row["record"])}

    def update_job(self, job_id, record, call_id=None):
        with self.transaction() as db:
            db.execute("UPDATE jobs SET record=?, call_id=COALESCE(?, call_id) WHERE id=?",
                       (json.dumps(record, allow_nan=False), call_id, job_id))

    def save_architecture(self, owner, spec, view=None, record_id=None):
        record_id = str(uuid.UUID(record_id)) if record_id else str(uuid.uuid4())
        now = datetime.fromtimestamp(self.clock(), UTC).isoformat()
        with self.transaction() as db:
            existing = db.execute("SELECT owner, record FROM architectures WHERE id=?", (record_id,)).fetchone()
            if existing and existing["owner"] != owner:
                raise KeyError(record_id)
            count = db.execute("SELECT COUNT(*) FROM architectures WHERE owner=?", (owner,)).fetchone()[0]
            if not existing and count >= LIMITS["saved_architectures"]:
                raise LimitError("The demo saves up to 20 architectures. Export additional designs instead.")
            record = {"id": record_id, "spec": spec, "updated_at": now,
                      "created_at": json.loads(existing["record"])["created_at"] if existing else now}
            if view is not None:
                if not isinstance(view, dict) or len(json.dumps(view, allow_nan=False)) > 200_000:
                    raise ValueError("Canvas view is too large.")
                record["view"] = view
            db.execute("INSERT OR REPLACE INTO architectures VALUES (?, ?, ?, ?)",
                       (record_id, owner, self.clock(), json.dumps(record, allow_nan=False)))
        return record

    def architectures(self, owner):
        with self.transaction() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT record FROM architectures WHERE owner=? ORDER BY updated DESC", (owner,))]

    def cleanup(self):
        import shutil
        cutoff = self.clock() - LIMITS["retention_days"] * 86400
        with self.transaction() as db:
            expired = db.execute("SELECT id FROM jobs WHERE created<?", (cutoff,)).fetchall()
            for row in expired:
                shutil.rmtree(self.root / "artifacts" / row[0], ignore_errors=True)
            db.execute("DELETE FROM jobs WHERE created<?", (cutoff,))
            db.execute("DELETE FROM architectures WHERE updated<?", (cutoff,))
            db.execute("DELETE FROM events WHERE created<?", (self.clock() - 62 * 86400,))
            db.execute("DELETE FROM challenges WHERE expires<?", (self.clock(),))
            db.execute("DELETE FROM sessions WHERE expires<?", (self.clock(),))
