"""Small SQLite helpers. Every call opens and closes its own connection."""

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "notes.db"


def init_db() -> None:
    """Create the persistent notes table if necessary."""
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    tags TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )"""
            )


def insert_note(content: str, tags: str = "") -> int:
    """Insert a note using bound SQL parameters and return its generated ID."""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Note content must not be empty.")
    init_db()
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        with connection:
            cursor = connection.execute(
                "INSERT INTO notes (content, tags) VALUES (?, ?)",
                (content, tags),
            )
            return int(cursor.lastrowid)


def search_notes(query: str) -> list[sqlite3.Row]:
    """Search literal keywords in content or tags with SQL LIKE, newest first."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Search query must not be empty.")
    init_db()
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    pattern = f"%{escaped}%"
    with closing(sqlite3.connect(DB_PATH, timeout=10)) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """SELECT id, content, tags, created_at FROM notes
               WHERE content LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\'
               ORDER BY id DESC""",
            (pattern, pattern),
        ).fetchall()
