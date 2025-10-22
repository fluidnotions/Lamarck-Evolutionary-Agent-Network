"""
Reasoning pattern memory system for LEAN.

Stores cognitive strategies (<think> content) rather than outputs (<final> content).
This is Layer 3 of the three-layer architecture: evolving reasoning patterns.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
from datetime import datetime
import time
import re
from dotenv import load_dotenv
from lean.logger import get_logger

load_dotenv()

logger = get_logger(__name__)


def generate_reasoning_collection_name(role: str, agent_id: str) -> str:
    """Generate unique collection name for an agent's reasoning patterns.

    Args:
        role: Agent role (intro, body, conclusion)
        agent_id: Unique agent identifier

    Returns:
        Collection name string
    """
    # Sanitize for ChromaDB (alphanumeric + underscores)
    safe_role = re.sub(r'[^a-zA-Z0-9_]', '', role)
    safe_id = re.sub(r'[^a-zA-Z0-9_]', '', agent_id)

    return f"{safe_role}_agent_{safe_id}_reasoning"


class ReasoningMemory:
    """Manages reasoning pattern storage and retrieval for agents with inheritance support.

    **CRITICAL**: This stores HOW agents think (<think> content), NOT what they produce (<final> content).

    Reasoning patterns include:
    - situation: Context/task description
    - tactic: Approach used (brief summary)
    - reasoning: Full <think> content (cognitive trace)
    - score: Quality of resulting output
    - retrieval_count: How often this pattern was used
    - generation: When pattern was created
    - inherited_from: Parent reasoning pattern IDs
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./data/reasoning",
        inherited_reasoning: Optional[List[Dict]] = None,
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize reasoning memory manager.

        Args:
            collection_name: Unique collection name (e.g., "intro_agent_1_reasoning")
            persist_directory: ChromaDB storage location
            inherited_reasoning: Reasoning patterns inherited from parents (None for generation 0)
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
        self.max_retrieve = int(os.getenv("MAX_REASONING_RETRIEVE", "5"))

        # Track retrieval counts
        self.retrieval_counts = {}

        # Load inherited reasoning patterns if provided
        if inherited_reasoning:
            self._load_inherited_reasoning(inherited_reasoning)

    def _load_inherited_reasoning(self, inherited_reasoning: List[Dict]):
        """Load reasoning patterns inherited from parents.

        Args:
            inherited_reasoning: List of reasoning pattern dicts from parent compaction
                Each dict: {'reasoning': str, 'score': float, 'metadata': dict}
        """
        if not inherited_reasoning:
            return

        documents = []
        metadatas = []
        embeddings = []
        ids = []

        for i, pattern in enumerate(inherited_reasoning):
            reasoning_id = f"{self.collection_name}_inherited_{i}_{time.time()}"

            # Get or create metadata
            metadata = pattern.get('metadata', {})
            metadata['inherited'] = True  # Mark as inherited
            metadata['original_score'] = pattern['score']
            metadata['score'] = pattern['score']
            metadata['timestamp'] = metadata.get('timestamp', datetime.now().isoformat())
            metadata['retrieval_count'] = 0
            metadata['situation'] = pattern.get('situation', '')
            metadata['tactic'] = pattern.get('tactic', '')

            # Generate embedding from reasoning content (NOT output content)
            reasoning = pattern['reasoning']
            embedding = self.embedder.encode(reasoning).tolist()

            documents.append(reasoning)
            metadatas.append(metadata)
            embeddings.append(embedding)
            ids.append(reasoning_id)

        # Bulk insert inherited reasoning patterns
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
                ids=ids
            )

    def store_reasoning_pattern(
        self,
        reasoning: str,
        score: float,
        situation: str = "",
        tactic: str = "",
        metadata: Optional[Dict] = None
    ) -> str:
        """Store reasoning pattern (ALL reasoning traces, no threshold).

        **CRITICAL**: This stores the <think> content, NOT the <final> output.

        Args:
            reasoning: The <think> section content (cognitive trace)
            score: LLM evaluation score of resulting output (0-10)
            situation: Task/context description
            tactic: Brief description of approach used
            metadata: Additional context (topic, generation, etc.)

        Returns:
            Reasoning pattern ID
        """
        # Generate embedding from reasoning content
        embedding = self.embedder.encode(reasoning).tolist()

        # Create unique reasoning pattern ID
        pattern_id = f"{self.collection_name}_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

        # Prepare full metadata
        full_metadata = {
            "score": score,
            "timestamp": time.time(),
            "generation": metadata.get("generation", 0) if metadata else 0,
            "topic": metadata.get("topic", "") if metadata else "",
            "domain": metadata.get("domain", "") if metadata else "",
            "situation": situation,
            "tactic": tactic,
            "inherited": False,  # Personal reasoning pattern
            "retrieval_count": 0,
            **(metadata or {})
        }

        # STORE EVERYTHING - no threshold
        self.collection.add(
            documents=[reasoning],  # The <think> content
            metadatas=[full_metadata],
            embeddings=[embedding],
            ids=[pattern_id]
        )

        return pattern_id

    def retrieve_similar_reasoning(
        self,
        query: str,
        k: int = 5,
        score_weight: float = 0.5
    ) -> List[Dict]:
        """Retrieve semantically similar reasoning patterns with score weighting.

        **Purpose**: Find "How did I/my parents solve similar problems?"

        Args:
            query: Query text (e.g., current task description)
            k: Number of reasoning patterns to retrieve
            score_weight: How much to weight by score (0=pure similarity, 1=balanced)

        Returns:
            List of dicts with 'reasoning', 'score', 'similarity', 'weighted_relevance', 'metadata'
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
            logger.warning(f"Reasoning retrieval error: {e}")
            return []

        if not results['documents'][0]:
            return []

        # Re-rank by weighted relevance
        weighted_results = []
        for i, doc in enumerate(results['documents'][0]):
            pattern_id = results['ids'][0][i]
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
            self.retrieval_counts[pattern_id] = self.retrieval_counts.get(pattern_id, 0) + 1

            # Increment in database
            self._increment_retrieval_count(pattern_id)

            weighted_results.append({
                'reasoning': doc,  # The <think> content
                'score': metadata['score'],
                'similarity': similarity,
                'weighted_relevance': weighted_relevance,
                'metadata': metadata,
                'retrieval_count': self.retrieval_counts[pattern_id],
                'id': pattern_id,
                'situation': metadata.get('situation', ''),
                'tactic': metadata.get('tactic', '')
            })

        # Sort by weighted relevance and return top K
        weighted_results.sort(key=lambda x: x['weighted_relevance'], reverse=True)
        return weighted_results[:k]

    def get_all_reasoning(self, include_inherited: bool = True) -> List[Dict]:
        """Export all reasoning patterns for compaction/analysis.

        Args:
            include_inherited: Whether to include inherited patterns

        Returns:
            List of all reasoning pattern dicts
        """
        # Handle empty collection
        if self.collection.count() == 0:
            return []

        # Get all reasoning patterns from collection
        all_data = self.collection.get(
            include=['documents', 'metadatas']
        )

        patterns = []
        for i, doc in enumerate(all_data['documents']):
            metadata = all_data['metadatas'][i]

            # Filter inherited if requested
            if not include_inherited and metadata.get('inherited', False):
                continue

            pattern_id = all_data['ids'][i]
            patterns.append({
                'id': pattern_id,
                'reasoning': doc,  # The <think> content
                'score': metadata['score'],
                'metadata': metadata,
                'retrieval_count': self.retrieval_counts.get(pattern_id, metadata.get('retrieval_count', 0)),
                'situation': metadata.get('situation', ''),
                'tactic': metadata.get('tactic', '')
            })

        return patterns

    def get_personal_reasoning(self) -> List[Dict]:
        """Get only personal (non-inherited) reasoning patterns.

        Returns:
            List of personal reasoning pattern dicts
        """
        return self.get_all_reasoning(include_inherited=False)

    def _increment_retrieval_count(self, pattern_id: str):
        """Increment the retrieval count for a reasoning pattern in database."""
        try:
            # Get current metadata
            result = self.collection.get(ids=[pattern_id], include=["metadatas"])
            if result["metadatas"]:
                metadata = result["metadatas"][0]
                metadata["retrieval_count"] = metadata.get("retrieval_count", 0) + 1

                # Update metadata
                self.collection.update(ids=[pattern_id], metadatas=[metadata])
        except Exception as e:
            logger.warning(f"Failed to increment retrieval count: {e}")

    def count(self) -> int:
        """Get total number of reasoning patterns stored."""
        return self.collection.count()

    def clear(self):
        """Clear all reasoning patterns from this collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.retrieval_counts = {}

    def get_stats(self) -> Dict:
        """Get statistics about this reasoning pattern collection."""
        count = self.count()
        if count == 0:
            return {
                "total_patterns": 0,
                "inherited_patterns": 0,
                "personal_patterns": 0,
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
            "total_patterns": count,
            "inherited_patterns": inherited_count,
            "personal_patterns": count - inherited_count,
            "collection": self.collection_name,
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "total_retrievals": sum(retrievals),
        }

    # Legacy compatibility alias for retrieve_similar_reasoning
    def retrieve_similar(self, query: str, k: int = 5, score_weight: float = 0.5) -> List[Dict]:
        """Alias for retrieve_similar_reasoning (compatibility)."""
        return self.retrieve_similar_reasoning(query, k, score_weight)

    # Alias for store_reasoning_pattern
    def store_experience(self, reasoning: str, score: float, metadata: Optional[Dict] = None) -> str:
        """Alias for store_reasoning_pattern (compatibility).

        Note: 'situation' and 'tactic' extracted from metadata if present.
        """
        situation = metadata.get('situation', '') if metadata else ''
        tactic = metadata.get('tactic', '') if metadata else ''
        return self.store_reasoning_pattern(reasoning, score, situation, tactic, metadata)
