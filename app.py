import os
import streamlit as st
from pdf_utils import load_pdf, split_text
from embedding_utils import embed_texts
from faiss_utils import (
    create_faiss_index, save_faiss_index, save_chunks, faiss_index_exists
)
from rag_pipeline import search_and_retrieve, ask_gemini

os.makedirs("data", exist_ok=True)

st.set_page_config(page_title="Gemini 1.5 Flash Search Genius", page_icon="📄")
st.title("📄 Search Genius - Document Q&A (Gemini 1.5 Flash)")

uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    if not faiss_index_exists():
        st.write("Processing uploaded document...")
        text = load_pdf(uploaded_file)
        chunks = split_text(text)
        embeddings = embed_texts(chunks)
        index = create_faiss_index(embeddings)
        save_faiss_index(index)
        save_chunks(chunks)
        st.success("Document indexed successfully!")
    else:
        st.warning("Index already exists. Delete files in /data folder to re-upload new file.")

if faiss_index_exists():
    question = st.text_input("Ask a question from the document:")
    if question:
        retrieved_chunks = search_and_retrieve(question)
        answer = ask_gemini(question, retrieved_chunks)
        st.write("### Answer:")
        st.write(answer)

        with st.expander("Show retrieved context chunks"):
            for i, chunk in enumerate(retrieved_chunks):
                st.write(f"**Chunk {i+1}:** {chunk}")
