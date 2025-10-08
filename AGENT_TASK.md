# Agent Task: Memory System

## Branch: `feature/memory-system`

## Priority: HIGH - Core HVAS concept

## Execution: PARALLEL with feature/state-management

## Objective
Implement the RAG (Retrieval-Augmented Generation) memory system using ChromaDB for vector storage and sentence-transformers for embeddings.

## Dependencies
- ✅ feature/project-foundation (must be merged first)

## Tasks

### 1. Create `src/hvas_mini/memory.py`

Implement memory management utilities:

```python
"""
Memory management system for HVAS Mini.

Handles ChromaDB operations, embeddings, and memory retrieval.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from datetime import datetime
import json
from dotenv import load_dotenv

load_dotenv()


class MemoryManager:
    """Manages memory storage and retrieval for agents."""

    def __init__(
        self,
        collection_name: str,
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: Optional[str] = None,
        persist_directory: str = "./data/memories"
    ):
        """Initialize memory manager.

        Args:
            collection_name: Name of the ChromaDB collection
            chroma_client: Optional existing ChromaDB client
            embedding_model: Name of sentence-transformers model
            persist_directory: Where to persist ChromaDB data
        """
        self.collection_name = collection_name

        # Initialize ChromaDB
        if chroma_client is None:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chroma_client

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        model_name = embedding_model or os.getenv(
            "EMBEDDING_MODEL",
            "all-MiniLM-L6-v2"
        )
        self.embedder = SentenceTransformer(model_name)

        # Configuration
        self.score_threshold = float(os.getenv("MEMORY_SCORE_THRESHOLD", "7.0"))
        self.max_retrieve = int(os.getenv("MAX_MEMORIES_RETRIEVE", "3"))

    def store(
        self,
        content: str,
        topic: str,
        score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store content in memory if score meets threshold.

        Args:
            content: The content to store
            topic: Associated topic
            score: Quality score (0-10)
            metadata: Additional metadata to store

        Returns:
            Memory ID if stored, empty string if below threshold
        """
        if score < self.score_threshold:
            return ""

        # Generate embedding
        embedding = self.embedder.encode(content).tolist()

        # Create memory ID
        memory_id = f"{self.collection_name}_{datetime.now().timestamp()}"

        # Prepare metadata
        meta = {
            "topic": topic,
            "score": score,
            "timestamp": datetime.now().isoformat(),
            "retrieval_count": 0,
            **(metadata or {})
        }

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding],
            documents=[content],
            metadatas=[meta],
            ids=[memory_id]
        )

        return memory_id

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        score_filter: Optional[float] = None
    ) -> List[Dict]:
        """Retrieve similar memories.

        Args:
            query: Query text to find similar memories
            n_results: Number of results (default from config)
            score_filter: Minimum score threshold (default from config)

        Returns:
            List of memory dictionaries with content, score, and metadata
        """
        n_results = n_results or self.max_retrieve
        score_filter = score_filter or self.score_threshold

        # Handle empty collection
        if self.collection.count() == 0:
            return []

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results, self.collection.count()),
                where={"score": {"$gte": score_filter}}
            )
        except Exception as e:
            print(f"[Warning] Memory retrieval error: {e}")
            return []

        # Handle no results
        if not results["documents"][0]:
            return []

        # Format results
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0.0

            memories.append({
                "content": doc,
                "topic": metadata.get("topic", ""),
                "score": metadata.get("score", 0.0),
                "timestamp": metadata.get("timestamp", ""),
                "distance": distance,
                "id": results["ids"][0][i]
            })

        # Sort by score (descending)
        memories.sort(key=lambda x: x["score"], reverse=True)

        # Increment retrieval counts
        for memory in memories:
            self._increment_retrieval_count(memory["id"])

        return memories

    def _increment_retrieval_count(self, memory_id: str):
        """Increment the retrieval count for a memory."""
        try:
            # Get current metadata
            result = self.collection.get(ids=[memory_id], include=["metadatas"])
            if result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + 1

                # Update metadata
                self.collection.update(
                    ids=[memory_id],
                    metadatas=[metadata]
                )
        except Exception as e:
            print(f"[Warning] Failed to increment retrieval count: {e}")

    def count(self) -> int:
        """Get total number of memories stored."""
        return self.collection.count()

    def clear(self):
        """Clear all memories from this collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict:
        """Get statistics about this memory collection."""
        count = self.count()
        if count == 0:
            return {
                "total_memories": 0,
                "collection": self.collection_name
            }

        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        scores = [m.get("score", 0) for m in all_data["metadatas"]]
        retrievals = [m.get("retrieval_count", 0) for m in all_data["metadatas"]]

        return {
            "total_memories": count,
            "collection": self.collection_name,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "total_retrievals": sum(retrievals),
            "score_threshold": self.score_threshold
        }
```

### 2. Create Tests

Create `test_memory.py`:

```python
"""Tests for memory system."""

from hvas_mini.memory import MemoryManager
import tempfile
import shutil
import pytest


@pytest.fixture
def temp_dir():
    """Create temporary directory for ChromaDB."""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def memory_manager(temp_dir):
    """Create memory manager for testing."""
    return MemoryManager(
        collection_name="test_collection",
        persist_directory=temp_dir
    )


def test_memory_storage(memory_manager):
    """Test storing memories."""
    memory_id = memory_manager.store(
        content="Machine learning is fascinating",
        topic="ML basics",
        score=8.5
    )

    assert memory_id != ""
    assert memory_manager.count() == 1


def test_memory_threshold(memory_manager):
    """Test score threshold filtering."""
    # Below threshold (default 7.0)
    memory_id = memory_manager.store(
        content="Low quality content",
        topic="test",
        score=5.0
    )

    assert memory_id == ""
    assert memory_manager.count() == 0


def test_memory_retrieval(memory_manager):
    """Test retrieving similar memories."""
    # Store some memories
    memory_manager.store(
        content="Machine learning uses algorithms to learn from data",
        topic="ML",
        score=9.0
    )
    memory_manager.store(
        content="Deep learning is a subset of machine learning",
        topic="ML",
        score=8.5
    )

    # Retrieve
    results = memory_manager.retrieve("what is machine learning")

    assert len(results) > 0
    assert "machine learning" in results[0]["content"].lower()
    assert results[0]["score"] >= 7.0


def test_memory_stats(memory_manager):
    """Test memory statistics."""
    memory_manager.store("Content 1", "topic", 8.0)
    memory_manager.store("Content 2", "topic", 9.0)

    stats = memory_manager.get_stats()

    assert stats["total_memories"] == 2
    assert stats["avg_score"] == 8.5
```

## Deliverables Checklist

- [ ] `src/hvas_mini/memory.py` with:
  - [ ] `MemoryManager` class
  - [ ] `store()` method with threshold filtering
  - [ ] `retrieve()` method with similarity search
  - [ ] `get_stats()` for analytics
  - [ ] Complete docstrings
- [ ] `test_memory.py` with passing tests
- [ ] ChromaDB persistence configuration
- [ ] Sentence-transformers integration

## Acceptance Criteria

1. ✅ Can store content in ChromaDB collections
2. ✅ Score threshold filtering works (stores only high-quality)
3. ✅ Semantic similarity retrieval works
4. ✅ Retrieval count tracking functions
5. ✅ All tests pass: `uv run pytest test_memory.py`
6. ✅ Persists data to `./data/memories`
7. ✅ Configuration loaded from .env

## Testing

```bash
cd worktrees/memory-system
uv run pytest test_memory.py -v
```

## Integration Notes

This memory system will be used by:
- `BaseAgent` for storing and retrieving memories
- Each specialized agent gets its own collection
- Collections persist across runs for learning

## Next Steps

After completion, merge to main and proceed with:
- feature/base-agent (will use MemoryManager)
