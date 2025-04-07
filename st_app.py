import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from ingestion_utils import extract_text
from rag import RAGChain
from vectorstore import ChromaDBClient

# ---- Einstellungen ----
st.set_page_config(page_title="Lernplattform: Audio Upload", layout="wide")

# ---- App Titel ----
st.title("🎓 Lernplattform – Audio Upload, Zusammenfassung & Q&A")
st.markdown(
    "Lade eine Audiodatei hoch. Das System transkribiert sie, \
        speichert sie in einer Vektordatenbank und beantwortet deine Fragen dazu."
)


# ---- Vektor-Datenbank und LLM ----
@st.cache_resource
def initialize_rag_chain():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = ChromaDBClient("learning_docs", "./chroma_db", embedding_model)
    llm = OllamaLLM(model="gemma3:1b")
    return RAGChain(vector_store, llm, k=3), vector_store


rag_chain, vector_store = initialize_rag_chain()

# ---- Audio Upload ----
audio_file = st.file_uploader("📤 Audiodatei hochladen (MP3/WAV)", type=["mp3", "wav"])

if audio_file is not None:
    temp_path = f"temp_{audio_file.name.replace(' ', '_')}"
    with open(temp_path, "wb") as f:
        f.write(audio_file.read())

    st.audio(temp_path)
    st.info("🔍 Transkription läuft...")

    transcript = extract_text(temp_path, model_size="base")
    st.subheader("📝 Transkript")
    st.text_area("Transkribierter Text", transcript, height=200)

    # ---- In Vektordatenbank speichern ----
    vector_store.add_document(audio_file.name.replace(" ", "_"), transcript)
    st.success("✅ Transkript in die Vektordatenbank eingefügt.")

# ---- Fragen stellen ----
st.subheader("❓ Stelle eine Frage zum Transkript")
user_question = st.text_input("Deine Frage:")

if user_question:
    st.info("🔎 Antwort wird gesucht...")
    answer = rag_chain.run(user_question)
    st.subheader("💡 Antwort")
    st.write(answer)
