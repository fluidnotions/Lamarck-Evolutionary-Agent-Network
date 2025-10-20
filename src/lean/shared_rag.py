"""
Shared RAG (Retrieval-Augmented Generation) system for LEAN.

Stores domain knowledge/facts available to ALL agents.
This is Layer 2 of the three-layer architecture: shared knowledge base.

**CRITICAL DISTINCTION**:
- ReasoningMemory (Layer 3): HOW to think - per-agent, evolves, inherited
- SharedRAG (Layer 2): WHAT to know - shared by all, fixed, not inherited
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class SharedRAG:
    """Shared knowledge base for domain facts, content, and references.

    All agents query the same shared RAG for domain knowledge.
    Only high-quality content (score ≥ 8.0) is stored here.

    This is separate from reasoning patterns - it stores:
    - Facts and references (from web search, documents)
    - High-quality outputs (from successful generations)
    - Domain-specific context and examples
    """

    def __init__(
        self,
        collection_name: str = "shared_knowledge",
        persist_directory: str = "./data/shared_rag",
        chroma_client: Optional[chromadb.Client] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize shared RAG manager.

        Args:
            collection_name: Collection name for shared knowledge
            persist_directory: ChromaDB storage location
            chroma_client: Optional existing ChromaDB client
            embedding_model: Name of sentence-transformers model
        """
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
        self.max_retrieve = int(os.getenv("MAX_KNOWLEDGE_RETRIEVE", "3"))
        self.min_quality_score = float(os.getenv("SHARED_RAG_MIN_SCORE", "8.0"))

    def store(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        source: str = "generated"
    ) -> str:
        """Store knowledge in shared RAG.

        Args:
            content: Knowledge content (facts, high-quality output, references)
            metadata: Additional context (topic, domain, score, etc.)
            source: Source of knowledge ("generated", "web_search", "manual")

        Returns:
            Knowledge ID
        """
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()

        # Create unique knowledge ID
        knowledge_id = f"knowledge_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

        # Prepare metadata
        full_metadata = {
            "source": source,
            "timestamp": time.time(),
            "topic": metadata.get("topic", "") if metadata else "",
            "domain": metadata.get("domain", "") if metadata else "",
            "score": metadata.get("score", 0.0) if metadata else 0.0,
            **(metadata or {})
        }

        # Store knowledge
        self.collection.add(
            documents=[content],
            metadatas=[full_metadata],
            embeddings=[embedding],
            ids=[knowledge_id]
        )

        return knowledge_id

    def store_if_high_quality(
        self,
        content: str,
        score: float,
        metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """Store content only if it meets quality threshold.

        **Purpose**: Only add high-quality content to shared knowledge base.

        Args:
            content: Content to potentially store
            score: Quality score (0-10)
            metadata: Additional context

        Returns:
            Knowledge ID if stored, None if below threshold
        """
        if score >= self.min_quality_score:
            meta = metadata or {}
            meta['score'] = score
            return self.store(content, meta, source="generated")
        return None

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        domain_filter: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant knowledge for a query.

        **Purpose**: Find "What facts/context do I need?"

        Args:
            query: Query text
            k: Number of results (defaults to max_retrieve)
            domain_filter: Optional domain filter (ML, Python, Web, etc.)

        Returns:
            List of dicts with 'content', 'metadata', 'similarity'
        """
        # Handle empty collection
        if self.collection.count() == 0:
            return []

        k = k or self.max_retrieve

        # Build filter
        where_filter = None
        if domain_filter:
            where_filter = {"domain": domain_filter}

        # Query ChromaDB
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter
            )
        except Exception as e:
            print(f"[Warning] Shared RAG retrieval error: {e}")
            return []

        if not results['documents'][0]:
            return []

        # Format results
        knowledge_items = []
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i] if results['distances'] else 0.0
            similarity = 1 - distance

            knowledge_items.append({
                'content': doc,
                'metadata': metadata,
                'similarity': similarity,
                'id': results['ids'][0][i],
                'topic': metadata.get('topic', ''),
                'domain': metadata.get('domain', ''),
                'source': metadata.get('source', 'unknown')
            })

        return knowledge_items

    def store_web_search_results(
        self,
        query: str,
        results: List[Dict],
        topic: str = "",
        domain: str = ""
    ) -> List[str]:
        """Store web search results (e.g., from Tavily API).

        Args:
            query: Search query
            results: List of search result dicts with 'content' or 'text' field
            topic: Topic associated with search
            domain: Domain category

        Returns:
            List of knowledge IDs
        """
        stored_ids = []

        for result in results:
            # Extract content from various result formats
            content = result.get('content') or result.get('text') or result.get('snippet', '')

            if not content:
                continue

            metadata = {
                'topic': topic,
                'domain': domain,
                'search_query': query,
                'url': result.get('url', ''),
                'title': result.get('title', ''),
            }

            knowledge_id = self.store(content, metadata, source="web_search")
            stored_ids.append(knowledge_id)

        return stored_ids

    def count(self) -> int:
        """Get total number of knowledge items stored."""
        return self.collection.count()

    def clear(self):
        """Clear all knowledge from shared RAG."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_stats(self) -> Dict:
        """Get statistics about shared knowledge base."""
        count = self.count()
        if count == 0:
            return {
                "total_knowledge": 0,
                "collection": self.collection_name,
                "by_source": {},
                "by_domain": {}
            }

        # Get all metadata
        all_data = self.collection.get(include=["metadatas"])

        # Count by source
        by_source = {}
        for m in all_data["metadatas"]:
            source = m.get("source", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

        # Count by domain
        by_domain = {}
        for m in all_data["metadatas"]:
            domain = m.get("domain", "general")
            if domain:  # Only count non-empty domains
                by_domain[domain] = by_domain.get(domain, 0) + 1

        return {
            "total_knowledge": count,
            "collection": self.collection_name,
            "by_source": by_source,
            "by_domain": by_domain,
        }

    def get_all_knowledge(self, source_filter: Optional[str] = None) -> List[Dict]:
        """Export all knowledge for analysis.

        Args:
            source_filter: Optional filter by source ("generated", "web_search", etc.)

        Returns:
            List of all knowledge dicts
        """
        # Handle empty collection
        if self.collection.count() == 0:
            return []

        # Get all knowledge
        all_data = self.collection.get(include=['documents', 'metadatas'])

        knowledge_items = []
        for i, doc in enumerate(all_data['documents']):
            metadata = all_data['metadatas'][i]

            # Filter by source if requested
            if source_filter and metadata.get('source') != source_filter:
                continue

            knowledge_items.append({
                'id': all_data['ids'][i],
                'content': doc,
                'metadata': metadata,
                'source': metadata.get('source', 'unknown'),
                'topic': metadata.get('topic', ''),
                'domain': metadata.get('domain', '')
            })

        return knowledge_items
