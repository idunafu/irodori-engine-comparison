"""SQLite WAL checkpoints: one committed outcome per request and scoring model."""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from common import sha256


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / 'results.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
          PRAGMA journal_mode=WAL;
          PRAGMA synchronous=FULL;
          CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY, spec TEXT NOT NULL, case_id TEXT NOT NULL,
            dataset TEXT NOT NULL, profile TEXT NOT NULL, mode TEXT NOT NULL,
            pilot INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'pending', result TEXT,
            wav TEXT, wav_sha256 TEXT);
          CREATE INDEX IF NOT EXISTS pending ON tasks(profile,mode,pilot,case_id,status);
          CREATE TABLE IF NOT EXISTS scores (
            task_id TEXT NOT NULL, scorer TEXT NOT NULL, wav_sha256 TEXT,
            status TEXT NOT NULL, result TEXT NOT NULL,
            PRIMARY KEY(task_id,scorer));
        ''')

    def populate(self, enumeration):
        batch=[]
        for key, spec, pilot in enumeration:
            batch.append((key,json.dumps(spec,ensure_ascii=False),spec['case']['id'],spec['item']['dataset'],
                          spec['case']['profile'],spec['mode'],int(pilot)))
            if len(batch)>=1000:
                self.db.executemany('INSERT OR IGNORE INTO tasks(id,spec,case_id,dataset,profile,mode,pilot) VALUES(?,?,?,?,?,?,?)',batch)
                self.db.commit();batch=[]
        self.db.executemany('INSERT OR IGNORE INTO tasks(id,spec,case_id,dataset,profile,mode,pilot) VALUES(?,?,?,?,?,?,?)',batch)
        self.db.commit()

    def outcome(self,key,status,result,wav=None):
        self.db.execute('DELETE FROM scores WHERE task_id=?',(key,))
        self.db.execute('UPDATE tasks SET status=?,result=?,wav=?,wav_sha256=? WHERE id=?',
            (status,json.dumps(result,ensure_ascii=False,allow_nan=False),str(wav) if wav else None,
             sha256(wav) if wav else None,key))
        self.db.commit()

    def score(self,key,scorer,status,result,wav_hash):
        self.db.execute('INSERT OR REPLACE INTO scores VALUES(?,?,?,?,?)',
            (key,scorer,wav_hash,status,json.dumps(result,ensure_ascii=False,allow_nan=False)))
        self.db.commit()

    def verify(self):
        bad=[]
        for row in self.db.execute("SELECT id,wav,wav_sha256 FROM tasks WHERE status='ok'"):
            if not row['wav'] or not Path(row['wav']).exists() or sha256(row['wav'])!=row['wav_sha256']:
                bad.append(row['id'])
        if bad:
            raise ValueError(f'Completed audio changed or missing ({len(bad)}): {bad[:5]}')
        stale=self.db.execute('''SELECT s.task_id FROM scores s JOIN tasks t ON s.task_id=t.id
                                WHERE s.wav_sha256 IS NOT t.wav_sha256 LIMIT 1''').fetchone()
        if stale:
            raise ValueError('Scoring cache does not match its generated audio')

    def counts(self):
        return dict(self.db.execute('SELECT status,COUNT(*) FROM tasks GROUP BY status').fetchall())
