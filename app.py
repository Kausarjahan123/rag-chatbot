import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# -----------------------------
# MODEL
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="AI RAG Chatbot", layout="wide")

st.title("AI Document Chatbot (RAG System)")

# -----------------------------
# SAMPLE DOCUMENT
# -----------------------------
text = """
Machine Learning is a field of AI that enables systems to learn from data.
Python is widely used in AI development.
Streamlit is used to build web apps quickly.
FAISS is used for similarity search in vector databases.
"""

# -----------------------------
# SPLIT TEXT INTO CHUNKS
# -----------------------------
chunks = text.split(".")

chunks = [c.strip() for c in chunks if c.strip()]

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = model.encode(chunks)

# FAISS INDEX
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# -----------------------------
# USER INPUT
# -----------------------------
query = st.text_input("Ask something from the document:")

# -----------------------------
# SEARCH FUNCTION
# -----------------------------
def search(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=2)
    return [chunks[i] for i in I[0]]

# -----------------------------
# RESPONSE
# -----------------------------
if query:
    results = search(query)

    st.subheader("Relevant Context")
    for r in results:
        st.write("-", r)

    st.subheader("Answer")
    st.write("Based on the document: ", results[0])
