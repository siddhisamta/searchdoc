import numpy as np
import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)

# 
# def embed_texts(texts):
#     embeddings = EMBEDDING_MODEL.embed_documents(texts)
#     return np.array(embeddings)

# def embed_query(query):
#     embedding = EMBEDDING_MODEL.embed_query(query)
#     return np.array([embedding])

import numpy as np

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text documents using the configured embedding model.

    Args:
        texts (List[str]): A list of input text documents to embed.

    Returns:
        np.ndarray: A NumPy array containing the embeddings for each input document.
    """

    # Generate embeddings for each document in the list using the embedding model
    document_embeddings: list[list[float]] = EMBEDDING_MODEL.embed_documents([str(t) for t in texts])

    # Convert the list of embeddings to a NumPy array for further processing
    return np.array(document_embeddings)


def embed_query(query: str) -> np.ndarray:
    """
    Generates an embedding for a single query string using the configured embedding model.

    Args:
        query (str): The input query text to embed.

    Returns:
        np.ndarray: A NumPy array containing the embedding vector for the query.
    """

    # Generate embedding for the query using the embedding model
    query_embedding: list[float] = EMBEDDING_MODEL.embed_query(str(query))

    # Return the embedding wrapped in a NumPy array for compatibility with vector operations
    return np.array([query_embedding])
