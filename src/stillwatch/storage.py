from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path

from .config import Settings
from .types import DecisionResult, EvidencePacket, Incident, Lifecycle


class Storage:
    SCHEMA_VERSION = 1

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root = settings.data_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.evidence = self.root / "incidents"
        self.evidence.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.error: str | None = None
        self.evidence_error: str | None = None
        self.db = sqlite3.connect(self.root / "stillwatch.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, self.SCHEMA_VERSION):
            raise ValueError("Unsupported storage schema version")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS incidents (
                id TEXT PRIMARY KEY, updated REAL NOT NULL, body TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, created REAL NOT NULL,
                packet TEXT NOT NULL, result TEXT NOT NULL,
                FOREIGN KEY(incident_id) REFERENCES incidents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS packets (
                id TEXT PRIMARY KEY, incident_id TEXT NOT NULL, created REAL NOT NULL,
                metadata TEXT NOT NULL, files TEXT NOT NULL,
                FOREIGN KEY(incident_id) REFERENCES incidents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY, incident_id TEXT NOT NULL, created REAL NOT NULL,
                operator TEXT NOT NULL, action TEXT NOT NULL,
                FOREIGN KEY(incident_id) REFERENCES incidents(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS calls (
                id INTEGER PRIMARY KEY, provider TEXT NOT NULL, started REAL NOT NULL,
                reserved INTEGER NOT NULL, tokens INTEGER, complete INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS calls_started ON calls(started);
        """)
        self.db.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        self.db.commit()

    def close(self):
        with self.lock:
            self.db.close()

    def _write(self, sql: str, parameters=()) -> int | None:
        with self.lock:
            try:
                cursor = self.db.execute(sql, parameters)
                self.db.commit()
                self.error = None
                return cursor.lastrowid
            except (sqlite3.Error, OSError):
                self.db.rollback()
                self.error = "metadata_write_failed"
                return None

    def save_incident(self, incident: Incident):
        self._write(
            "INSERT INTO incidents(id,updated,body) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated,body=excluded.body",
            (incident.incident_id, incident.updated_at, incident.model_dump_json()),
        )

    def incidents(self) -> list[Incident]:
        with self.lock:
            return [
                Incident.model_validate_json(row[0])
                for row in self.db.execute("SELECT body FROM incidents ORDER BY updated DESC")
            ]

    def save_packet(self, packet: EvidencePacket) -> bool:
        files = []
        missing = False
        with self.lock:
            used = self.evidence_bytes()
            needed = sum(len(i.jpeg) for i in packet.images)
            if used + needed > self.settings.storage_quota_bytes:
                self.cleanup(make_room=needed)
                used = self.evidence_bytes()
            if used + needed > self.settings.storage_quota_bytes:
                missing = True
                self.evidence_error = "evidence_quota_exhausted"
            else:
                folder = self.evidence / packet.incident_id / packet.packet_id
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    for index, image in enumerate(packet.images):
                        path = folder / f"{index}.jpg"
                        temporary = path.with_suffix(".part")
                        temporary.write_bytes(image.jpeg)
                        temporary.replace(path)
                        files.append(str(path.relative_to(self.evidence)))
                    self.evidence_error = None
                except OSError:
                    missing = True
                    self.evidence_error = "evidence_write_failed"
            self._write(
                "INSERT OR REPLACE INTO packets VALUES (?,?,?,?,?)",
                (
                    packet.packet_id,
                    packet.incident_id,
                    time.time(),
                    json.dumps(packet.metadata()),
                    json.dumps(files),
                ),
            )
        return not missing

    def save_result(self, packet: EvidencePacket, result: DecisionResult):
        self._write(
            "INSERT INTO assessments(incident_id,created,packet,result) VALUES (?,?,?,?)",
            (packet.incident_id, time.time(), json.dumps(packet.metadata()), result.model_dump_json()),
        )

    def feedback(self, incident_id: str, operator: str, action: str):
        self._write(
            "INSERT INTO feedback(incident_id,created,operator,action) VALUES (?,?,?,?)",
            (incident_id, time.time(), operator, action),
        )

    def detail(self, incident_id: str) -> dict:
        with self.lock:
            packets = []
            for row in self.db.execute(
                "SELECT * FROM packets WHERE incident_id=? ORDER BY created", (incident_id,)
            ):
                meta = json.loads(row["metadata"])
                meta["files"] = [
                    {
                        "url": f"/evidence/{incident_id}/{row['id']}/{i}",
                        "available": (self.evidence / f).is_file(),
                    }
                    for i, f in enumerate(json.loads(row["files"]))
                ]
                packets.append(meta)
            return {
                "packets": packets,
                "assessments": [
                    json.loads(r[0])
                    for r in self.db.execute(
                        "SELECT result FROM assessments WHERE incident_id=? ORDER BY created", (incident_id,)
                    )
                ],
                "feedback_history": [
                    dict(r)
                    for r in self.db.execute(
                        "SELECT created,operator,action FROM feedback WHERE incident_id=? ORDER BY created",
                        (incident_id,),
                    )
                ],
            }

    def evidence_path(self, incident_id: str, packet_id: str, index: int) -> Path | None:
        if not all(re.fullmatch(r"[a-f0-9]{32}", identifier) for identifier in (incident_id, packet_id)):
            return None
        with self.lock:
            row = self.db.execute(
                "SELECT files FROM packets WHERE id=? AND incident_id=?", (packet_id, incident_id)
            ).fetchone()
        if not row:
            return None
        files = json.loads(row[0])
        if not 0 <= index < len(files):
            return None
        path = (self.evidence / files[index]).resolve()
        return path if path.is_relative_to(self.evidence.resolve()) and path.is_file() else None

    def reserve_call(self, provider: str, reserved: int, now: float) -> int | None:
        return self._write(
            "INSERT INTO calls(provider,started,reserved) VALUES (?,?,?)", (provider, now, reserved)
        )

    def finish_call(self, call_id: int, tokens: int | None):
        self._write("UPDATE calls SET tokens=?,complete=1 WHERE id=?", (tokens, call_id))

    def calls(self, provider: str, since: float) -> list[dict]:
        with self.lock:
            return [
                dict(r)
                for r in self.db.execute(
                    "SELECT * FROM calls WHERE provider=? AND started>=?", (provider, since)
                )
            ]

    def evidence_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.evidence.rglob("*") if p.is_file())

    def cleanup(self, now: float | None = None, make_room: int = 0):
        now = now or time.time()
        with self.lock:
            try:
                cutoff = now - self.settings.frame_retention_hours * 3600
                for path in self.evidence.rglob("*"):
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                # Quota pressure evicts closed material first; open event evidence is retained until expiry.
                used = self.evidence_bytes()
                for incident in sorted(self.incidents(), key=lambda i: i.updated_at):
                    if used + make_room <= self.settings.storage_quota_bytes:
                        break
                    folder = self.evidence / incident.incident_id
                    if incident.lifecycle == Lifecycle.CLOSED and folder.is_dir():
                        size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
                        shutil.rmtree(folder)
                        used -= size
                expiry = now - self.settings.metadata_retention_days * 86400
                expired = [
                    r[0] for r in self.db.execute("SELECT id FROM incidents WHERE updated<?", (expiry,))
                ]
                for identifier in expired:
                    folder = self.evidence / identifier
                    if folder.is_dir() and folder.resolve().is_relative_to(self.evidence.resolve()):
                        shutil.rmtree(folder)
                self._write("DELETE FROM incidents WHERE updated<?", (expiry,))
                self._write("DELETE FROM calls WHERE started<?", (now - 7 * 86400,))
            except OSError:
                self.error = "cleanup_failed"

    def stats(self) -> dict:
        try:
            free = shutil.disk_usage(self.root).free
            used = self.evidence_bytes()
        except OSError:
            free, used = 0, 0
        return {
            "evidence_bytes": used,
            "quota_bytes": self.settings.storage_quota_bytes,
            "disk_free_bytes": free,
            "error": self.error or self.evidence_error,
        }
