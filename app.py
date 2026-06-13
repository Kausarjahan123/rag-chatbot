import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pypdf import PdfReader

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Document Copilot", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

h1 {
    text-align: center;
    color: #4da3ff;
}

.block {
    background: #161b22;
    padding: 12px;
    border-radius: 10px;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Document Copilot SaaS")

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# SESSION STATE (CHAT MEMORY)
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "index" not in st.session_state:
    st.session_state.index = None

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # CLEAN CHUNKING
    chunks = [c.strip() for c in text.split("\n") if len(c.strip()) > 40]

    # EMBEDDINGS
    embeddings = model.encode(chunks)

    # VECTOR DB
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return chunks, index

# -----------------------------
# PROCESS FILE
# -----------------------------
if uploaded_file:
    chunks, index = process_pdf(uploaded_file)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("Document processed successfully")

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve(query, chunks, index, k=4):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]

# -----------------------------
# ANSWER ENGINE
# -----------------------------
def generate_answer(query, context):
    context_text = "\n".join(context)

    return f"""
Based on your document:

{context_text}

Final Answer:
This response is generated using a Retrieval-Augmented Generation (RAG) system.
It combines semantic search + context understanding to answer: {query}
"""

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Ask anything from your document")

if query and st.session_state.index:

    chunks = st.session_state.chunks
    index = st.session_state.index

    context = retrieve(query, chunks, index)

    answer = generate_answer(query, context)

    st.session_state.chat.append((query, answer))

# -----------------------------
# CHAT UI
# -----------------------------
for q, a in st.session_state.chat:
    st.markdown("---")
    st.markdown(f"**You:** {q}")
    st.markdown(f"**AI:** {a}")

# -----------------------------
# SIDEBAR (SAAS INSIGHTS)
# -----------------------------
st.sidebar.title("Document Insights")

if st.session_state.chunks:
    st.sidebar.write("Chunks:", len(st.session_state.chunks))
    st.sidebar.write("Status: Active")
    st.sidebar.write("System: RAG + FAISS + Transformers")
