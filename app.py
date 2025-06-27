import os
import streamlit as st
from pdf_utils import load_pdf, split_text
from embedding_utils import embed_texts
from faiss_utils import (
    create_faiss_index, save_faiss_index, save_chunks, faiss_index_exists
)
from rag_pipeline import search_and_retrieve, ask_gemini

os.makedirs("data", exist_ok=True)

# st.set_page_config(page_title="Gemini 1.5 Flash Search Genius", page_icon="📄")
# st.title("📄 Search Genius - Document Q&A (Gemini 1.5 Flash)")

# uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

# if uploaded_file is not None:
#     if not faiss_index_exists():
#         st.write("Processing uploaded document...")
#         text = load_pdf(uploaded_file)
#         chunks = split_text(text)
#         embeddings = embed_texts(chunks)
#         index = create_faiss_index(embeddings)
#         save_faiss_index(index)
#         save_chunks(chunks)
#         st.success("Document indexed successfully!")
#     else:
#         st.warning("Index already exists. Delete files in /data folder to re-upload new file.")

# if faiss_index_exists():
#     question = st.text_input("Ask a question from the document:")
#     if question:
#         retrieved_chunks = search_and_retrieve(question)
#         answer = ask_gemini(question, retrieved_chunks)
#         st.write("### Answer:")
#         st.write(answer)

#         with st.expander("Show retrieved context chunks"):
#             for i, chunk in enumerate(retrieved_chunks):
#                 st.write(f"**Chunk {i+1}:** {chunk}")

# Configure the Streamlit page
st.set_page_config(page_title="Gemini 1.5 Flash Search Genius", page_icon="📄")
st.title("📄 Search Genius - Document Q&A (Gemini 1.5 Flash)")

# File uploader to upload PDF files
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    # Check if index already exists to avoid reprocessing
    if not faiss_index_exists():
        st.write("Processing uploaded document...")

        # Step 1: Extract text from PDF
        document_text: str = load_pdf(uploaded_file)

        # Step 2: Split text into manageable chunks
        document_chunks: list[str] = split_text(document_text)

        # Step 3: Convert text chunks to embeddings
        chunk_embeddings = embed_texts(document_chunks)

        # Step 4: Create FAISS index from embeddings
        faiss_index = create_faiss_index(chunk_embeddings)

        # Step 5: Save index and text chunks locally
        save_faiss_index(faiss_index)
        save_chunks(document_chunks)

        st.success("Document indexed successfully!")
    else:
        st.warning("Index already exists. Delete files in /data folder to re-upload a new file.")

# Input for asking questions from the uploaded document
if faiss_index_exists():
    user_question: str = st.text_input("Ask a question from the document:")

    if user_question:
        # Step 1: Retrieve relevant chunks using FAISS
        relevant_chunks: list[str] = search_and_retrieve(user_question)

        # Step 2: Pass question and context to Gemini model
        ai_response: str = ask_gemini(user_question, relevant_chunks)

        # Display the answer
        st.write("### Answer:")
        st.write(ai_response)

        # Optional: Show retrieved chunks for transparency
        with st.expander("Show retrieved context chunks"):
            for i, chunk in enumerate(relevant_chunks):
                st.write(f"**Chunk {i + 1}:** {chunk}")