"""
Memory management system for HVAS Mini.

Handles ChromaDB operations, embeddings, and memory retrieval with weighted scoring.
Supports memory inheritance for evolutionary knowledge transfer.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from datetime import datetime
import json
import time
import re
from dotenv import load_dotenv

load_dotenv()


def generate_collection_name(role: str, agent_id: str) -> str:
    """Generate unique collection name for an agent.

    Args:
        role: Agent role (intro, body, conclusion)
        agent_id: Unique agent identifier

    Returns:
        Collection name string
    """
    # Sanitize for ChromaDB (alphanumeric + underscores)
    safe_role = re.sub(r'[^a-zA-Z0-9_]', '', role)
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '', agent_id)

    return f"{safe_role}_agent_{safe_id}_memories"


class MemoryManager:
    """Manages memory storage and retrieval for agents with inheritance support."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/memories",
        inherited_memories: Optional[List[Dict]] = None,
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize memory manager.

        Args:
            collection_name: Unique collection name (e.g., "intro_agent_1_memories")
            persist_directory: ChromaDB storage location
            inherited_memories: Memories inherited from parents (None for generation 0)
            chroma_client: Optional existing ChromaDB client
            embedding_model: Name of sentence-transformers model
        """
        # Validate collection name
        if not re.match(r'^[a-zA-Z0-9_]+$', collection_name):
            raise ValueError(f"Invalid collection name: {collection_name}")

        self.collection_name = collection_name
        self.persist_directory = persist_directory

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
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)

        # Configuration
        self.max_retrieve = int(os.getenv("MAX_MEMORIES_RETRIEVE", "5"))

        # Track retrieval counts
        self.retrieval_counts = {}

        # Load inherited memories if provided
        if inherited_memories:
            self._load_inherited_memories(inherited_memories)

    def _load_inherited_memories(self, inherited_memories: List[Dict]):
        """Load memories inherited from parents.

        Args:
            inherited_memories: List of memory dicts from parent compaction
                Each dict: {'content': str, 'score': float, 'metadata': dict}
        """
        if not inherited_memories:
            return

        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for i, mem in enumerate(inherited_memories):
            memory_id = f"{self.collection_name}_inherited_{i}_{time.time()}"

            # Get or create metadata
            metadata = mem.get('metadata', {})
            metadata['inherited'] = True  # Mark as inherited
            metadata['original_score'] = mem['score']
            metadata['score'] = mem['score']
            metadata['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
            metadata['retrieval_count'] = 0

            # Generate embedding
            content = mem['content']
            embedding = self.embedder.encode(content).tolist()

            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)
            ids.append(memory_id)

        # Bulk insert inherited memories
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )

    def store_experience(
        self,
        output: str,
        score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Store agent output (ALL experiences, no threshold).

        Args:
            output: Generated content
            score: LLM evaluation score (0-10)
            metadata: Additional context (topic, generation, etc.)

        Returns:
            Memory ID
        """
        # Generate embedding
        embedding = self.embedder.encode(output).tolist()

        # Create unique memory ID
        memory_id = f"{self.collection_name}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

        # Prepare full metadata
        full_metadata = {
            "score": score,
            "timestamp": time.time(),
            "generation": metadata.get("generation", 0) if metadata else 0,
            "topic": metadata.get("topic", "") if metadata else "",
            "domain": metadata.get("domain", "") if metadata else "",
            "inherited": False,  # Personal memory
            "retrieval_count": 0,
            **(metadata or {})
        }

        # STORE EVERYTHING - no threshold
        self.collection.add(
            documents=[output],
            metadatas=[full_metadata],
            embeddings=[embedding],
            ids=[memory_id]
        )

        return memory_id

    def retrieve_similar(
        self,
        query: str,
        k: int = 5,
        score_weight: float = 0.5
    ) -> List[Dict]:
        """Retrieve semantically similar memories with score weighting.

        Args:
            query: Query text (e.g., current task description)
            k: Number of memories to retrieve
            score_weight: How much to weight by score (0=pure similarity, 1=balanced)

        Returns:
            List of dicts with 'content', 'score', 'similarity', 'weighted_relevance'
        """
        # Handle empty collection
        if self.collection.count() == 0:
            return []

        # Query ChromaDB for semantic similarity
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k * 3, 50)  # Get extra, then re-rank
            )
        except Exception as e:
            print(f"[Warning] Memory retrieval error: {e}")
            return []

        if not results['documents'][0]:
            return []

        # Re-rank by weighted relevance
        weighted_results = []
        for i, doc in enumerate(results['documents'][0]):
            memory_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if results['distances'] else 0.0
            similarity = 1 - distance  # cosine distance → similarity

            # Weighted relevance: combine semantic similarity + quality score
            score_factor = (metadata['score'] / 10.0)  # Normalize to 0-1
            weighted_relevance = (
                (1 - score_weight) * similarity +
                score_weight * (similarity * score_factor)
            )

            # Track retrieval for frequency metrics
            self.retrieval_counts[memory_id] = self.retrieval_counts.get(memory_id, 0) + 1

            # Increment in database
            self._increment_retrieval_count(memory_id)

            weighted_results.append({
                'content': doc,
                'score': metadata['score'],
                'similarity': similarity,
                'weighted_relevance': weighted_relevance,
                'metadata': metadata,
                'retrieval_count': self.retrieval_counts[memory_id],
                'id': memory_id
            })

        # Sort by weighted relevance and return top K
        weighted_results.sort(key=lambda x: x['weighted_relevance'], reverse=True)
        return weighted_results[:k]

    def get_all_memories(self, include_inherited: bool = True) -> List[Dict]:
        """Export all memories for compaction/analysis.

        Args:
            include_inherited: Whether to include inherited memories

        Returns:
            List of all memory dicts with content, score, metadata
        """
        # Handle empty collection
        if self.collection.count() == 0:
            return []

        # Get all memories from collection
        all_data = self.collection.get(
            include=['documents', 'metadatas']
        )

        memories = []
        for i, doc in enumerate(all_data['documents']):
            metadata = all_data['metadatas'][i]

            # Filter inherited if requested
            if not include_inherited and metadata.get('inherited', False):
                continue

            memory_id = all_data['ids'][i]
            memories.append({
                'id': memory_id,
                'content': doc,
                'score': metadata['score'],
                'metadata': metadata,
                'retrieval_count': self.retrieval_counts.get(memory_id, metadata.get('retrieval_count', 0))
            })

        return memories

    def get_personal_memories(self) -> List[Dict]:
        """Get only personal (non-inherited) memories.

        Returns:
            List of personal memory dicts
        """
        return self.get_all_memories(include_inherited=False)

    # Legacy compatibility methods (for existing code)
    def store(
        self,
        content: str,
        topic: str,
        score: float,
        metadata: Optional[Dict] = None
    ) -> str:
        """Legacy store method for compatibility.

        Args:
            content: The content to store
            topic: Associated topic
            score: Quality score (0-10)
            metadata: Additional metadata to store

        Returns:
            Memory ID
        """
        meta = metadata or {}
        meta['topic'] = topic
        return self.store_experience(content, score, meta)

    def _increment_retrieval_count(self, memory_id: str):
        """Increment the retrieval count for a memory in database."""
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
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.retrieval_counts = {}

    def get_stats(self) -> Dict:
        """Get statistics about this memory collection."""
        count = self.count()
        if count == 0:
            return {
                "total_memories": 0,
                "inherited_memories": 0,
                "personal_memories": 0,
                "collection": self.collection_name
            }

        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])
        scores = [m.get("score", 0) for m in all_data["metadatas"]]
        retrievals = [
            self.retrieval_counts.get(all_data['ids'][i], m.get("retrieval_count", 0))
            for i, m in enumerate(all_data["metadatas"])
        ]
        inherited_count = sum(1 for m in all_data["metadatas"] if m.get("inherited", False))

        return {
            "total_memories": count,
            "inherited_memories": inherited_count,
            "personal_memories": count - inherited_count,
            "collection": self.collection_name,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "total_retrievals": sum(retrievals),
        }

    def retrieve(
        self,
        query: str,
        n_results: Optional[int] = None,
        score_filter: Optional[float] = None,
    ) -> List[Dict]:
        """Legacy retrieve method for compatibility.

        Args:
            query: Query text to find similar memories
            n_results: Number of results
            score_filter: Minimum score threshold (DEPRECATED - not used)

        Returns:
            List of memory dictionaries with content, score, and metadata
        """
        k = n_results or self.max_retrieve
        results = self.retrieve_similar(query, k=k, score_weight=0.5)

        # Convert to legacy format
        return [
            {
                "content": r['content'],
                "topic": r['metadata'].get('topic', ''),
                "score": r['score'],
                "timestamp": r['metadata'].get('timestamp', ''),
                "distance": 1 - r['similarity'],
                "similarity": r['similarity'],
                "effective_score": r['weighted_relevance'],
                "id": r['id'],
            }
            for r in results
        ]
