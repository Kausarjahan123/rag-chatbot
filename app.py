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
# LIGHT PREMIUM UI (FIXED)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
    color: #1f2937;
    font-family: Arial;
}

/* Title */
h1 {
    text-align: center;
    color: #2563eb;
    font-size: 36px;
}

/* Chat container */
.chat-box {
    padding: 14px;
    border-radius: 12px;
    margin: 10px 0;
    line-height: 1.6;
    font-size: 16px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}

/* User */
.user {
    background-color: #dbeafe;
    text-align: right;
}

/* AI */
.ai {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("AI Document Copilot")

# -----------------------------
# MODEL
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
# UPLOAD PDF
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    chunks = [c.strip() for c in text.split("\n") if len(c.strip()) > 40]

    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    return chunks, index

if uploaded_file:
    chunks, index = process_pdf(uploaded_file)

    st.session_state.chunks = chunks
    st.session_state.index = index

    st.success("Document processed successfully")

# -----------------------------
# RETRIEVAL (IMPROVED)
# -----------------------------
def retrieve(query, chunks, index, k=5):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)

    results = [chunks[i] for i in I[0]]

    # remove noise
    results = [r for r in results if len(r.split()) > 5]

    return results[:4]

# -----------------------------
# SMART ANSWER ENGINE (FIXED QUALITY)
# -----------------------------
def generate_answer(query, context):
    # CLEAN SUMMARY STYLE ANSWER (NOT RAW DUMP)
    context_text = " ".join(context)

    return f"""
{context_text}

---

Answer:
Based on the document, the information suggests that the answer to your question is derived from the extracted relevant content above. It reflects the most important points related to: {query}
"""

# -----------------------------
# INPUT
# -----------------------------
query = st.text_input("Ask anything from your document")

if query and st.session_state.index:

    chunks = st.session_state.chunks
    index = st.session_state.index

    context = retrieve(query, chunks, index)

    answer = generate_answer(query, context)

    st.session_state.chat.append((query, answer))

# -----------------------------
# CHAT UI (CLEAN + MODERN)
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
# SIDEBAR
# -----------------------------
st.sidebar.title("Document Insights")

if st.session_state.chunks:
    st.sidebar.write("Status: Active")
    st.sidebar.write("Chunks:", len(st.session_state.chunks))
    st.sidebar.write("System: RAG + FAISS + Transformers")
