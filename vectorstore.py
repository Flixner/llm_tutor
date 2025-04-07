import logging
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ChromaDBClient:
    """Client for interacting with a ChromaDB vector database."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: HuggingFaceEmbeddings,
    ) -> None:
        """Initializes the ChromaDB client.

        Args:
            collection_name: Name of the ChromaDB collection.
            persist_directory: Directory to persist database.
            embedding_model: Embedding model instance.
        """
        self._embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=persist_directory)
        try:
            self._collection = self._client.get_collection(name=collection_name)
        except Exception:
            self._collection = self._client.create_collection(name=collection_name)

    def _ensure_list_float(self, embedding) -> list[float]:
        """Ensure embedding is a float list."""
        if isinstance(embedding, list):
            return embedding
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return list(map(float, embedding))

    def add_document(self, filename: str, content: str) -> None:
        """Add a single document to the collection."""
        embedding = self._embedding_model.embed_documents([content])[0]
        embedding = self._ensure_list_float(embedding)

        self._collection.add(ids=[filename], documents=[content], embeddings=[embedding])

    def add_documents_from_directory(self, directory: str) -> None:
        """Read all files from a directory and add them."""
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Invalid directory: {directory}")

        try:
            existing_ids = set(self._collection.get()["ids"])
        except Exception:
            existing_ids = set()

        for file_path in dir_path.iterdir():
            if not file_path.is_file():
                continue
            doc_id = file_path.name
            if doc_id in existing_ids:
                logger.info("Document %s already exists. Skipping.", doc_id)
                continue

            content = file_path.read_text(encoding="utf-8")
            self.add_document(filename=doc_id, content=content)
            logger.info("Added document: %s", doc_id)

    def query(
        self,
        query_text: str,
        k: int = 5,
        included_info: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return top-k matching documents' IDs and distances."""
        if not query_text.strip():
            raise ValueError("Query text must not be empty.")

        included_info = included_info or ["distances"]
        embedding = self._embedding_model.embed_query(query_text)
        embedding = self._ensure_list_float(embedding)

        results = self._collection.query(query_embeddings=[embedding], n_results=k, include=included_info)

        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in included_info else [0.0] * len(ids)

        return list(zip(ids, distances))

    def retrieve_documents(
        self,
        query_text: str,
        k: int = 5,
        distance_threshold: float | None = None,
    ) -> list[str]:
        """Retrieve the top-k documents optionally filtered by distance."""
        if not query_text.strip():
            raise ValueError("Query text must not be empty.")

        embedding = self._embedding_model.embed_query(query_text)
        embedding = self._ensure_list_float(embedding)

        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if distance_threshold is not None:
            return [doc for doc, dist in zip(documents, distances) if dist < distance_threshold]

        return documents


if __name__ == "__main__":
    collection_name: str = "learning_docs"
    persist_directory: str = "./chroma_db"

    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    client = ChromaDBClient(collection_name, persist_directory, model)

    client.add_documents_from_directory("./data/transcriptions")

    query_text: str = "Was ist das logische Datenmodell"
    top_docs = client.retrieve_documents(query_text, k=3)
    print("Top matching documents (doc_id, distance):", top_docs)
