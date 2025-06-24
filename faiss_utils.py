import faiss
import numpy as np
import pickle
import os

DATA_DIR = "data"
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.index")
CHUNK_DATA_FILE = os.path.join(DATA_DIR, "chunk_data.pkl")

def create_faiss_index(embeddings):
    embeddings = np.array(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index

def save_faiss_index(index):
    faiss.write_index(index, FAISS_INDEX_FILE)

def load_faiss_index():
    return faiss.read_index(FAISS_INDEX_FILE)

def save_chunks(chunks):
    with open(CHUNK_DATA_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_chunks():
    with open(CHUNK_DATA_FILE, "rb") as f:
        return pickle.load(f)

def faiss_index_exists():
    return os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNK_DATA_FILE)
