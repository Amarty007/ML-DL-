"""Small SQLite data store for datasets used by the Streamlit app."""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "data" / "ml_dl_playground.db"


def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                csv_data BLOB NOT NULL,
                row_count INTEGER NOT NULL,
                column_names TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_dataset(file_name, csv_data, dataframe):
    """Save a dataset once and return its database id."""
    content_hash = hashlib.sha256(csv_data).hexdigest()
    with _connect() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO datasets
                (file_name, content_hash, csv_data, row_count, column_names, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                file_name,
                content_hash,
                sqlite3.Binary(csv_data),
                len(dataframe),
                json.dumps(list(dataframe.columns)),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        row = connection.execute(
            "SELECT id FROM datasets WHERE content_hash = ?", (content_hash,)
        ).fetchone()
    return row["id"]


def list_datasets():
    with _connect() as connection:
        return connection.execute(
            """
            SELECT id, file_name, row_count, column_names, created_at
            FROM datasets
            ORDER BY created_at DESC, id DESC
            """
        ).fetchall()


def load_dataset(dataset_id):
    with _connect() as connection:
        row = connection.execute(
            "SELECT file_name, csv_data FROM datasets WHERE id = ?", (dataset_id,)
        ).fetchone()
    if row is None:
        raise ValueError("Selected dataset was not found in the database.")
    return row["file_name"], bytes(row["csv_data"])


def delete_dataset(dataset_id):
    with _connect() as connection:
        connection.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
