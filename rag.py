from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

from vectorstore import ChromaDBClient


class RAGChain:
    """Retrieval-Augmented Generation (RAG) Chain for a local personal tutor.

    This chain retrieves relevant document context from the vector store and uses a local LLM
    (GPT4All) to generate an answer based on that context.
    """

    def __init__(self, vector_store: ChromaDBClient, llm, k: int = 3) -> None:
        """
        Args:
            vector_store: Instance of ChromaDBClient containing indexed documents.
            llm: A local GPT4All instance for generating responses.
            k: Number of top documents to retrieve.
        """
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
        """Generates an answer for the given question using retrieved document context.

        Args:
            question: The user's question.

        Returns:
            A generated answer as a string.
        """
        # Retrieve relevant documents from the vector store.
        results = self.vector_store._collection.query(
            query_embeddings=[self.vector_store._embedding_model.embed_query(question)],
            n_results=self.k,
            include=["documents", "distances"],
        )
        # Extract document texts from the query results.
        documents: list[str] = results.get("documents", [[]])[0]
        context: str = "\n".join(documents)
        # Format the prompt with the retrieved context and the user question.
        prompt: str = self.prompt_template.format(context=context, question=question)
        # Generate an answer using the local GPT4All model.
        answer: str = self.llm.invoke(prompt)
        return answer


if __name__ == "__main__":
    # Initialize the vector store (from your vectorstore.py).
    collection_name: str = "learning_docs"
    persist_directory: str = "./chroma_db"
    vector_store = ChromaDBClient(collection_name, persist_directory)

    # Define the GPT4All model path; update this to your local GPT4All model file.
    # llm_model_path: str = "./Phi-3-mini-4k-instruct-q4.gguf"
    # model_path_obj = Path(llm_model_path)
    # if not model_path_obj.is_file():
    #    raise FileNotFoundError(
    #        f"GPT4All model file not found at {llm_model_path}. "
    #        "Please update the path to your local GPT4All model file."
    #    )

    # Initialize the local GPT4All LLM.
    llm = OllamaLLM(model="gemma3:1b")

    # Create the RAG chain instance.
    rag_chain = RAGChain(vector_store, llm, k=3)

    # Example query.
    query_text: str = """
    What is the data model? Answer only based on the documents you got. " \
    "If there is no information in the document tell me. Answer in german
    """
    answer: str = rag_chain.run(query_text)
    print("Answer:", answer)
