"""
Memory Store - Hybrid search: 0.7 x vector + 0.3 x keyword (BM25)
This is where the agent's memories live.  Each memory is stored with:
-   The text context
-   A category (decision, preference, lesson, fact, context) 
-   A vector embedding (for semantic search)
-   Time stamps
   """

import sqlite3
import datetime
import re
from pathlib import Path

from agent.memory.embeddings import (
    embed_text,
    serialize_vector,
    deserialize_vector,
    cosine_similarity,
)

# Hybrid search weights - from Cole Medlin's architecture
VECTOR_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

class MemoryStore:
    """
    Persistent memory backed by SQLite.
    
    Usage:
        store = MemoryStore(Path("./memory/agent.db"), Path("./memory"))
        store.save("User prefers dark mode", category="preference")
        results = store.search("what theme does the user like?")
    """
    def __init__(self, db_path: Path, memory_dir: Path):
        self.db_path = Path(db_path)
        self.memory_dir = Path(memory_dir)

        # make sure the directory for the database exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # create tables if they don't exist
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a database connection"""
        return sqlite3.connect(str(self.db_path))
    
    def _init_db(self):
        """Create the database tables if they don't exist"""
        conn = self._connect()

        conn.executescript("""
            -- Main table: stores memories with their embeddings
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'fact',
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
                           
            -- FTS5 virtual table for fast keyword search
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                content,
                category,
                content_rowid='id'
            );
                           
            -- Trigger: when we INSERT into memories, also insert into FTS
            CREATE TRIGGER IF NOT EXISTS memories_ai 
                AFTER INSERT ON memories 
            BEGIN
                INSERT INTO memories_fts(rowid, content, category)
                VALUES (new.id, new.content, new.category);
            END;  

            -- Trigger: when we UPDATE memories, update FTS too
            CREATE TRIGGER IF NOT EXISTS memories_au 
                AFTER UPDATE ON memories 
            BEGIN
                UPDATE memories_fts 
                SET content = new.content, category = new.category
                WHERE rowid = old.id;
            END;

            -- Trigger: when we DELETE from memories, delete from FTS too
            CREATE TRIGGER IF NOT EXISTS memories_ad 
                AFTER DELETE ON memories 
            BEGIN
                DELETE FROM memories_fts WHERE rowid = old.id;
            END;                                    
        """)

        conn.close()

    def save(self, content: str, category: str = "fact") -> int:
        """Save new memory"""
        now = datetime.datetime.now().isoformat()

        # Generate the embedding vector for this memory
        embedding = embed_text(content)
        embedding_blob = serialize_vector(embedding)

        conn = self._connect()
        cursor = conn.execute(
            """INSERT INTO memories
                (content, category, embedding, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)""",
                (content, category, embedding_blob, now, now)
        )
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()

        if memory_id is None:
            raise RuntimeError("Failed to save memory - no ID returned")
        return memory_id
    
    def search(self, query: str, limit: int=5) -> list[dict]:
        """Hybrid search: 0.7 x vector similarity + 0.3 x keyword BM25."""
        # Step 1: Get the query embedding vector
        query_embedding = embed_text(query)

        conn = self._connect()

        # Step 2: Load all memories for vector comparison
        rows = conn.execute(
            "SELECT id, content, category, embedding, created_at FROM memories"
        ).fetchall()

        if not rows:
            conn.close()
            return []
        
        # Step 3: Calculate vector similarity for each memory
        vector_scores = {}
        for row in rows:
            mem_id, content, category, emb_blob, created_at = row
            if emb_blob:
                mem_embedding = deserialize_vector(emb_blob)
                similarity = cosine_similarity(query_embedding, mem_embedding)
                vector_scores[mem_id] = {
                    "content": content,
                    "category": category,
                    "created_at": created_at,
                    "vector_score": max(0.0, similarity),
                }

        # Step 4: Run keyword search using FTS5       
        keyword_scores = {}
        try:
            # Remove special characters that break FTS5 queries
            safe_query = re.sub(r'[^\w\s]', '', query)

            if safe_query.strip():
                fts_rows = conn.execute(
                    """ SELECT rowid, rank
                        FROM memories_fts
                        WHERE memories_fts MATCH ?
                        ORDER BY rank
                        LIMIT ?""",
                    (safe_query, limit * 3)
                ).fetchall()

                if fts_rows:
                    min_rank = min(r[1] for r in fts_rows)
                    max_rank = max(r[1] for r in fts_rows)
                    rank_range = max_rank - min_rank if max_rank != min_rank else 1.0

                    for rowid, rank in fts_rows:
                        normalized = 1.0 - ((rank - min_rank) / rank_range) if rank_range else 1.0
                        keyword_scores[rowid] = normalized
        except Exception:
            pass # Keyword search failed; vector search still works 
        
        conn.close()

        # Step #5: Combine scores with weighted average
        combined = []
        for mem_id, data in vector_scores.items():
            v_score = data["vector_score"]
            k_score = keyword_scores.get(mem_id, 0.0)
            hybrid_score = (VECTOR_WEIGHT * v_score) + (KEYWORD_WEIGHT * k_score)

            combined.append({
                "id": mem_id,
                "content": data["content"],
                "category": data["category"],
                "created_at": data["created_at"],
                "score": hybrid_score,
                "vector_score": v_score,
                "keyword_score": k_score,
            })  
        # Step 6: Sort by hybrid score (highest first) and return top results
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:limit]

    def get_all(self, limit: int = 100) -> list[dict]:
        """Get all memories, most recent first"""
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, content, category, created_at FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()

        return [
            {id: r[0], "content": r[1], "category": r[2], "created_at": r[3]}
            for r in rows
        ]
    
    def delete(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        conn = self._connect()
        conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return affected > 0
    
    def count(self) -> int:
        """How many memories are stored."""
        conn = self._connect()
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return count
        
        
