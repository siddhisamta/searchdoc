import faiss
import numpy as np
import pickle
import os

DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
CHUNK_DATA_FILE = os.path.join(DATA_DIR, "chunk_data.pkl")

# def create_faiss_index(embeddings):
#     embeddings = np.array(embeddings)
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings.astype(np.float32))
#     return index

# def save_faiss_index(index):
#     faiss.write_index(index, FAISS_INDEX_FILE)

# def load_faiss_index():
#     return faiss.read_index(FAISS_INDEX_FILE)

# def save_chunks(chunks):
#     with open(CHUNK_DATA_FILE, "wb") as f:
#         pickle.dump(chunks, f)

# def load_chunks():
#     with open(CHUNK_DATA_FILE, "rb") as f:
#         return pickle.load(f)

# def faiss_index_exists():
#     return os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_DATA_FILE)

def create_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index from the provided document embeddings.

    Args:
        embeddings (list[list[float]]): A 2D list of embedding vectors.

    Returns:
        faiss.IndexFlatL2: A FAISS index built using L2 distance metric.
    """
    embeddings_array: np.ndarray = np.array(embeddings)
    dimension: int = embeddings_array.shape[1]

    # Create a FAISS index with L2 distance for the given embedding dimensions
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings to the index (must be float32 type)
    index.add(embeddings_array.astype(np.float32))
    return index


def save_faiss_index(index: faiss.IndexFlatL2) -> None:
    """
    Saves the FAISS index to disk.

    Args:
        index (faiss.IndexFlatL2): The FAISS index object to save.
    """
    faiss.write_index(index, FAISS_INDEX_FILE)


def load_faiss_index() -> faiss.IndexFlatL2:
    """
    Loads the FAISS index from disk.

    Returns:
        faiss.IndexFlatL2: The loaded FAISS index object.
    """
    return faiss.read_index(FAISS_INDEX_FILE)


def save_chunks(chunks: list[str]) -> None:
    """
    Saves the document chunks to a pickle file on disk.

    Args:
        chunks (list[str]): A list of document chunks (strings) to save.
    """
    with open(CHUNK_DATA_FILE, "wb") as f:
        pickle.dump(chunks, f)


def load_chunks() -> list[str]:
    """
    Loads the document chunks from disk.

    Returns:
        list[str]: The list of document chunks loaded from the pickle file.
    """
    with open(CHUNK_DATA_FILE, "rb") as f:
        return pickle.load(f)


def faiss_index_exists() -> bool:
    """
    Checks if both the FAISS index file and chunk data file exist.

    Returns:
        bool: True if both files exist, False otherwise.
    """
    return os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_DATA_FILE)