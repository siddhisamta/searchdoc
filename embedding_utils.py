import numpy as np
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# embeddings_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

def embed_texts(texts):
    embeddings = EMBEDDING_MODEL.embed_documents(texts)
    return np.array(embeddings)

def embed_query(query):
    embedding = EMBEDDING_MODEL.embed_query(query)
    return np.array([embedding])
