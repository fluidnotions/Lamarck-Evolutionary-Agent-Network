"""
Memory management system for HVAS Mini.

Handles ChromaDB operations, embeddings, and memory retrieval with time-based decay.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from datetime import datetime
import json
from dotenv import load_dotenv

# Import decay calculator
try:
    from hvas_mini.memory.decay import DecayCalculator
except ImportError:
    # For standalone development
    DecayCalculator = None

load_dotenv()


class MemoryManager:
    """Manages memory storage and retrieval for agents."""

    def __init__(
        self,
        collection_name: str,
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: Optional[str] = None,
        persist_directory: str = "./data/memories",
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
                path=persist_directory, settings=Settings(anonymized_telemetry=False)
            )
        else:
            self.client = chroma_client

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

        # Initialize embedding model
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)

        # Configuration
        self.score_threshold = float(os.getenv("MEMORY_SCORE_THRESHOLD", "7.0"))
        self.max_retrieve = int(os.getenv("MAX_MEMORIES_RETRIEVE", "3"))

        # NEW: Initialize decay calculator (M3)
        if DecayCalculator:
            decay_lambda = float(os.getenv("MEMORY_DECAY_LAMBDA", "0.01"))
            self.decay_calculator = DecayCalculator(decay_lambda=decay_lambda)
        else:
            self.decay_calculator = None

    def store(
        self, content: str, topic: str, score: float, metadata: Optional[Dict] = None
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
            **(metadata or {}),
        }

        # Store in ChromaDB
        self.collection.add(
            embeddings=[embedding], documents=[content], metadatas=[meta], ids=[memory_id]
        )

        return memory_id

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        score_filter: Optional[float] = None,
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
                where={"score": {"$gte": score_filter}},
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

            # NEW: Calculate effective score with decay (M3)
            similarity = 1.0 - distance  # Convert distance to similarity
            original_score = metadata.get("score", 0.0)
            timestamp = metadata.get("timestamp", datetime.now().isoformat())

            if self.decay_calculator:
                effective_score = self.decay_calculator.calculate_effective_score(
                    similarity=similarity,
                    original_score=original_score,
                    timestamp=timestamp,
                )
            else:
                # Fallback: no decay
                effective_score = similarity * (original_score / 10.0)

            memories.append(
                {
                    "content": doc,
                    "topic": metadata.get("topic", ""),
                    "score": original_score,
                    "timestamp": timestamp,
                    "distance": distance,
                    "similarity": similarity,
                    "effective_score": effective_score,
                    "id": results["ids"][0][i],
                }
            )

        # Sort by effective score (descending) - this incorporates decay
        memories.sort(key=lambda x: x["effective_score"], reverse=True)

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
                self.collection.update(ids=[memory_id], metadatas=[metadata])
        except Exception as e:
            print(f"[Warning] Failed to increment retrieval count: {e}")

    def count(self) -> int:
        """Get total number of memories stored."""
        return self.collection.count()

    def clear(self):
        """Clear all memories from this collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict:
        """Get statistics about this memory collection."""
        count = self.count()
        if count == 0:
            return {"total_memories": 0, "collection": self.collection_name}

        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        scores = [m.get("score", 0) for m in all_data["metadatas"]]
        retrievals = [m.get("retrieval_count", 0) for m in all_data["metadatas"]]

        return {
            "total_memories": count,
            "collection": self.collection_name,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "total_retrievals": sum(retrievals),
            "score_threshold": self.score_threshold,
        }
