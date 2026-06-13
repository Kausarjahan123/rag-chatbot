import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pypdf import PdfReader

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Document Copilot", layout="wide")

# -----------------------------
# CLEAN SAAS UI (FIXED)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #e5e7eb;
    font-family: Arial;
}

/* Title */
h1 {
    text-align: center;
    color: #60a5fa;
    font-size: 34px;
}

/* Chat boxes */
.chat-box {
    padding: 12px;
    border-radius: 10px;
    margin: 10px 0;
    line-height: 1.5;
}

/* User message */
.user {
    background-color: #1d4ed8;
    color: white;
    text-align: right;
}

/* AI message */
.ai {
    background-color: #1f2937;
    color: #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Document Copilot SaaS")

# -----------------------------
# MODEL (CACHE FOR SPEED)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# SESSION STATE
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
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    # SMART CHUNKING
    chunks = [c.strip() for c in text.split("\n") if len(c.strip()) > 40]

    # EMBEDDINGS
    embeddings = model.encode(chunks)

    # VECTOR DB (FAISS)
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
# RETRIEVAL ENGINE (IMPROVED)
# -----------------------------
def retrieve(query, chunks, index, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)

    results = [chunks[i] for i in I[0]]

    # remove noisy / too-short results
    results = [r for r in results if len(r.split()) > 5]

    return results[:4]

# -----------------------------
# ANSWER ENGINE (FIXED QUALITY)
# -----------------------------
def generate_answer(query, context):
    context_text = "\n".join(context)

    return f"""
Answer:

Based on your document, here is the most relevant information:

{context_text}

Final Explanation:
This answer is generated using semantic search (RAG system). It uses the most relevant parts of your document to respond accurately to: "{query}"
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
# CHAT UI (CLEAN + READABLE)
# -----------------------------
for q, a in st.session_state.chat:

    st.markdown(f"""
    <div class="chat-box user">
        <b>You:</b><br>{q}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="chat-box ai">
        <b>AI:</b><br>{a}
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# SIDEBAR (SAAS INSIGHTS)
# -----------------------------
st.sidebar.title("Document Insights")

if st.session_state.chunks:
    st.sidebar.write("Status: Active")
    st.sidebar.write("Chunks:", len(st.session_state.chunks))
    st.sidebar.write("System: RAG + FAISS + Transformers")
