"""SQLite is authoritative; JSONL files are replaceable ordered projections."""

import contextlib
import json
import sqlite3
import uuid
from pathlib import Path

from qps.util import atomic_json, atomic_text, canonical, file_lock, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
 id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, spec TEXT NOT NULL, resolved TEXT,
 status TEXT NOT NULL, phase TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0,
 cancel INTEGER NOT NULL DEFAULT 0, created TEXT NOT NULL, updated TEXT NOT NULL,
 owner TEXT, worker TEXT, elapsed_seconds REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS attempts (
 run_id TEXT NOT NULL REFERENCES runs(id), number INTEGER NOT NULL,
 started TEXT NOT NULL, ended TEXT, status TEXT NOT NULL, reason TEXT,
 PRIMARY KEY(run_id, number)
);
CREATE TABLE IF NOT EXISTS phases (
 run_id TEXT NOT NULL REFERENCES runs(id), attempt INTEGER NOT NULL, phase TEXT NOT NULL,
 status TEXT NOT NULL, output TEXT, PRIMARY KEY(run_id, attempt, phase)
);
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL REFERENCES runs(id),
 attempt INTEGER NOT NULL, time TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS artifacts (
 id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), kind TEXT NOT NULL,
 path TEXT NOT NULL UNIQUE, manifest_sha TEXT NOT NULL, created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS deployments (
 name TEXT PRIMARY KEY, artifact_id TEXT NOT NULL REFERENCES artifacts(id),
 state TEXT NOT NULL, identity TEXT, url TEXT, profile TEXT NOT NULL, log TEXT NOT NULL,
 created TEXT NOT NULL
);
PRAGMA user_version=1;
"""


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "metadata.sqlite"
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextlib.contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def run(self, name: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id=? OR name=?", (name, name)).fetchone()
        if not row:
            raise ValueError(f"Unknown run: {name}")
        return dict(row)

    def runs(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM runs ORDER BY created DESC")]

    def directory(self, run_id: str) -> Path:
        return self.root / "runs" / self.run(run_id)["id"]

    def create_run(self, spec, submitted: str | None = None, *, run_id: str | None = None) -> dict:
        run_id = run_id or "run-" + uuid.uuid4().hex
        if not __import__("re").fullmatch(r"run-[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run identity")
        data = spec.model_dump(mode="json", by_alias=True)
        try:
            with self.connect() as db:
                db.execute(
                    "INSERT INTO runs(id,name,spec,status,phase,created,updated) "
                    "VALUES(?,?,?,'pending','submitted',?,?)",
                    (run_id, spec.metadata.name, canonical(data), now(), now()),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Run name already exists: {spec.metadata.name}") from exc
        directory = self.directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        atomic_text(directory / "submitted.yaml", submitted or canonical(data))
        atomic_json(directory / "spec.json", data)
        self.emit(run_id, 0, "submitted", {"name": spec.metadata.name})
        return self.run(run_id)

    def update(self, run_id: str, **fields):
        allowed = {
            "status",
            "phase",
            "attempt",
            "cancel",
            "resolved",
            "owner",
            "worker",
            "elapsed_seconds",
        }
        if not fields.keys() <= allowed:
            raise ValueError("Invalid run update")
        fields["updated"] = now()
        with self.connect() as db:
            db.execute(
                "UPDATE runs SET " + ",".join(f"{key}=?" for key in fields) + " WHERE id=?",
                (*fields.values(), run_id),
            )

    def begin_attempt(self, run_id: str, owner: dict) -> int:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            attempt = db.execute("SELECT attempt FROM runs WHERE id=?", (run_id,)).fetchone()[0] + 1
            db.execute(
                "UPDATE runs SET attempt=?,status='running',cancel=0,owner=?,worker=NULL,updated=? "
                "WHERE id=?",
                (attempt, canonical(owner), now(), run_id),
            )
            db.execute(
                "INSERT INTO attempts VALUES(?,?,?,NULL,'running',NULL)", (run_id, attempt, now())
            )
        self.emit(run_id, attempt, "attempt_started", {})
        return attempt

    def finish(self, run_id: str, attempt: int, status: str, reason: str | None = None):
        with self.connect() as db:
            db.execute(
                "UPDATE attempts SET ended=?,status=?,reason=? WHERE run_id=? AND number=?",
                (now(), status, reason, run_id, attempt),
            )
            db.execute(
                "UPDATE runs SET status=?,worker=NULL,owner=NULL,updated=? WHERE id=?",
                (status, now(), run_id),
            )
            payload = {"status": status, "reason": reason}
            db.execute(
                "INSERT INTO events(run_id,attempt,time,kind,payload) VALUES(?,?,?,?,?)",
                (run_id, attempt, now(), "attempt_finished", canonical(payload)),
            )
        self.project(run_id)

    def phase(self, run_id: str, attempt: int, phase: str, status: str, output=None):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO phases VALUES(?,?,?,?,?)",
                (run_id, attempt, phase, status, canonical(output) if output is not None else None),
            )
            db.execute("UPDATE runs SET phase=?,updated=? WHERE id=?", (phase, now(), run_id))
            db.execute(
                "INSERT INTO events(run_id,attempt,time,kind,payload) VALUES(?,?,?,?,?)",
                (run_id, attempt, now(), "phase_" + status, canonical({"phase": phase})),
            )
        self.project(run_id)

    def output(self, run_id: str, phase: str):
        with self.connect() as db:
            row = db.execute(
                "SELECT output FROM phases WHERE run_id=? AND phase=? AND status='succeeded' "
                "ORDER BY attempt DESC LIMIT 1",
                (run_id, phase),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def emit(self, run_id: str, attempt: int, kind: str, payload: dict):
        encoded = canonical(payload)  # Reject non-finite metrics before writing either sink.
        with self.connect() as db:
            current = db.execute("SELECT attempt FROM runs WHERE id=?", (run_id,)).fetchone()
            if current is None or current[0] != attempt:
                raise RuntimeError("Stale attempt cannot publish events")
            db.execute(
                "INSERT INTO events(run_id,attempt,time,kind,payload) VALUES(?,?,?,?,?)",
                (run_id, attempt, now(), kind, encoded),
            )
        self.project(run_id)

    def events(self, run_id: str, after: int = 0):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id", (run_id, after)
            ).fetchall()
        return [{**dict(row), "payload": json.loads(row["payload"])} for row in rows]

    def project(self, run_id: str):
        directory = self.directory(run_id)
        with file_lock(directory / "projection.lock"):
            events = self.events(run_id)
            atomic_text(directory / "events.jsonl", "".join(canonical(e) + "\n" for e in events))
            metrics = [e for e in events if e["kind"] == "train_step"]
            atomic_text(directory / "metrics.jsonl", "".join(canonical(e) + "\n" for e in metrics))

    def artifact(self, artifact_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        if not row:
            raise ValueError(f"Unknown artifact: {artifact_id}")
        return dict(row)

    def artifacts(self, run_id: str):
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM artifacts WHERE run_id=? ORDER BY created", (run_id,)
                )
            ]

    def deployment(self, name: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM deployments WHERE name=?", (name,)).fetchone()
        if not row:
            raise ValueError(f"Unknown deployment: {name}")
        return dict(row)
