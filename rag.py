from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from vectorstore import ChromaDBClient


class RAGChain:
    """Retrieval-Augmented Generation (RAG) chain using ChromaDB and a local LLM."""

    def __init__(self, vector_store: ChromaDBClient, llm: OllamaLLM, k: int = 3) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a knowledgeable personal tutor. Use the following context to answer the question.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

    def run(self, question: str) -> str:
        """Generate an answer based on retrieved document context."""
        documents: list[str] = self.vector_store.retrieve_documents(question, k=self.k)
        if not documents:
            return "Keine Informationen in den Dokumenten gefunden."

        context: str = "\n".join(documents)
        prompt: str = self.prompt_template.format(context=context, question=question)
        answer: str = self.llm.invoke(prompt)
        return answer


if __name__ == "__main__":
    collection_name: str = "learning_docs"
    persist_directory: str = "./chroma_db"
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = ChromaDBClient(collection_name, persist_directory, embedding_model)

    llm = OllamaLLM(model="gemma3:1b")

    rag_chain = RAGChain(vector_store, llm, k=3)

    query_text: str = """
    What is the data model? Answer only based on the documents you got. " \
    "If there is no information in the document tell me. Answer in german
    """
    answer: str = rag_chain.run(query_text)
    print("Answer:", answer)
