import numpy as np
from embedding_utils import embed_query
from faiss_utils import load_faiss_index, load_chunks
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def search_and_retrieve(question, top_k=3):
    index = load_faiss_index()
    chunks = load_chunks()
    query_embedding = embed_query(question)
    D, I = index.search(query_embedding.astype(np.float32), top_k)
    retrieved_chunks = [chunks[i] for i in I[0]]
    return retrieved_chunks

def ask_gemini(question, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"Answer the question based on context:\n{context}\n\nQuestion: {question}"

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )

    response = llm.invoke(prompt)
    return response.content
