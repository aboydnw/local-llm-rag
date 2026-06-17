import hashlib
import json
import re
import sqlite3
import struct
from pathlib import Path

import sqlite_vec

from rag_lab.types import Chunk


def _vec_to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _to_fts_query(query: str) -> str:
    """Turn free-text into a safe FTS5 MATCH expression.

    User questions contain punctuation (``?``, ``-``, quotes) that FTS5 parses as
    query syntax. We extract bare word tokens, quote each as a phrase, and OR them
    so any keyword match contributes.
    """
    tokens = re.findall(r"\w+", query)
    return " OR ".join(f'"{token}"' for token in tokens)


def _chunk_id(chunk: Chunk) -> str:
    key = f"{chunk.doc_path}|{chunk.position}|{chunk.text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class SqliteVecStore:
    """sqlite-vec backed store for chunks and their embeddings."""

    def __init__(self, path: Path, dimension: int) -> None:
        self.path = path
        self.dimension = dimension

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_path TEXT NOT NULL,
                    heading_path TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    embedding FLOAT[{self.dimension}]
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS vec_ids (
                    rowid INTEGER PRIMARY KEY,
                    chunk_id TEXT NOT NULL UNIQUE
                )
                """
            )
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize = 'porter'
                )
                """
            )
            conn.commit()

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors length mismatch")
        with self._connect() as conn:
            for chunk, vector in zip(chunks, vectors, strict=True):
                cid = _chunk_id(chunk)
                conn.execute(
                    """
                    INSERT INTO chunks(id, doc_path, heading_path, position, text, metadata)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        doc_path=excluded.doc_path,
                        heading_path=excluded.heading_path,
                        position=excluded.position,
                        text=excluded.text,
                        metadata=excluded.metadata
                    """,
                    (
                        cid,
                        str(chunk.doc_path),
                        json.dumps(list(chunk.heading_path)),
                        chunk.position,
                        chunk.text,
                        json.dumps(chunk.metadata),
                    ),
                )
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
                conn.execute(
                    "INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)",
                    (cid, chunk.text),
                )
                row = conn.execute(
                    "SELECT rowid FROM vec_ids WHERE chunk_id = ?", (cid,)
                ).fetchone()
                if row is None:
                    cursor = conn.execute(
                        "INSERT INTO vec_ids(chunk_id) VALUES(?)", (cid,)
                    )
                    rowid = cursor.lastrowid
                    conn.execute(
                        "INSERT INTO vec_chunks(rowid, embedding) VALUES(?, ?)",
                        (rowid, _vec_to_blob(vector)),
                    )
                else:
                    rowid = row[0]
                    conn.execute(
                        "UPDATE vec_chunks SET embedding = ? WHERE rowid = ?",
                        (_vec_to_blob(vector), rowid),
                    )
            conn.commit()

    def count(self) -> int:
        with self._connect() as conn:
            (n,) = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(n)

    def query_bm25(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        match = _to_fts_query(query)
        if not match:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, rank
                FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (match, k),
            ).fetchall()
            out: list[tuple[Chunk, float]] = []
            for chunk_id, rank in rows:
                cdata = conn.execute(
                    "SELECT doc_path, heading_path, position, text, metadata "
                    "FROM chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                doc_path, heading_path_json, position, text, metadata_json = cdata
                out.append(
                    (
                        Chunk(
                            text=text,
                            doc_path=Path(doc_path),
                            heading_path=tuple(json.loads(heading_path_json)),
                            position=position,
                            metadata=json.loads(metadata_json),
                        ),
                        float(-rank),
                    )
                )
            return out

    def query_vector(self, vector: list[float], k: int) -> list[tuple[Chunk, float]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT vec_ids.chunk_id, v.distance
                FROM (
                    SELECT rowid, distance
                    FROM vec_chunks
                    WHERE embedding MATCH ? AND k = ?
                    ORDER BY distance
                ) AS v
                JOIN vec_ids ON vec_ids.rowid = v.rowid
                ORDER BY v.distance
                """,
                (_vec_to_blob(vector), k),
            ).fetchall()
            out: list[tuple[Chunk, float]] = []
            for chunk_id, distance in rows:
                cdata = conn.execute(
                    "SELECT doc_path, heading_path, position, text, metadata "
                    "FROM chunks WHERE id = ?",
                    (chunk_id,),
                ).fetchone()
                doc_path, heading_path_json, position, text, metadata_json = cdata
                chunk = Chunk(
                    text=text,
                    doc_path=Path(doc_path),
                    heading_path=tuple(json.loads(heading_path_json)),
                    position=position,
                    metadata=json.loads(metadata_json),
                )
                out.append((chunk, float(distance)))
            return out
