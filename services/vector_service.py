"""Vector service — ChromaDB for furniture product recommendations."""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from models.schemas import VectorMatch

logger = logging.getLogger(__name__)

_client: Optional[chromadb.ClientAPI] = None
_embedder: Optional[SentenceTransformer] = None

COLLECTION_NAME = "furniture_products"


def _get_client(path: str = "./chroma_store") -> chromadb.ClientAPI:
    global _client
    if _client is None:
        logger.info("Initializing ChromaDB at %s …", path)
        _client = chromadb.PersistentClient(path=path)
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading sentence-transformer model …")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def get_collection(path: str = "./chroma_store") -> chromadb.Collection:
    """Get or create the furniture_products collection."""
    client = _get_client(path)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def query_recommendations(
    labels: list[str],
    chroma_path: str = "./chroma_store",
    n_results: int = 5,
) -> list[VectorMatch]:
    """Query ChromaDB for furniture products related to detected labels.

    Args:
        labels: List of detected furniture/object labels.
        chroma_path: Path to ChromaDB persistence directory.
        n_results: Number of results to return.

    Returns:
        List of VectorMatch objects.
    """
    if not labels:
        return []

    collection = get_collection(chroma_path)

    # Check if collection has data
    if collection.count() == 0:
        logger.warning("ChromaDB collection is empty — run seed_vector_db.py first.")
        return []

    # Build a query string from detected labels
    query_text = ", ".join(labels)
    embedder = _get_embedder()
    query_embedding = embedder.encode(query_text).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches: list[VectorMatch] = []
    if results and results["ids"] and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            distance = results["distances"][0][i] if results["distances"] else 0.0
            similarity = max(0.0, 1.0 - distance)

            matches.append(
                VectorMatch(
                    id=doc_id,
                    name=meta.get("name", "Unknown"),
                    description=results["documents"][0][i] if results["documents"] else "",
                    category=meta.get("category", ""),
                    style=meta.get("style", ""),
                    room_type=meta.get("room_type", ""),
                    similarity_score=round(similarity, 3),
                )
            )

    logger.info("Vector search returned %d matches for labels: %s", len(matches), labels)
    return matches
