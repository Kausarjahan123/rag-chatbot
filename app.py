import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from pypdf import PdfReader
import os

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")
st.title("AI Python Notes Chatbot")

# -----------------------------
# PDF PATH (FIXED)
# -----------------------------
pdf_path = "Python.pdf"

# safety check (IMPORTANT)
if not os.path.exists(pdf_path):
    st.error(f"File not found: {pdf_path}. Please check filename in GitHub.")
    st.stop()

# -----------------------------
# LOAD PDF
# -----------------------------
def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

text = load_pdf_text(pdf_path)

# -----------------------------
# SPLIT INTO CHUNKS
# -----------------------------
chunks = text.split(".")
chunks = [c.strip() for c in chunks if len(c.strip()) > 10]

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = model.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=3)
    return [chunks[i] for i in I[0]]

# -----------------------------
# UI
# -----------------------------
query = st.text_input("Ask a question from your Python notes:")

if query:
    results = search(query)

    st.subheader("Relevant Content")
    for r in results:
        st.write("-", r)

    st.subheader("Answer (based on notes)")
    st.success(results[0])
