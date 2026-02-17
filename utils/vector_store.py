import os
from langchain_community.vectorstores import FAISS
from utils.embeddings import get_embeddings


def create_vector_store(chunks):
    embeddings = get_embeddings()
    return FAISS.from_texts(chunks, embedding=embeddings)


def save_vector_store(vector_store, path):
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def load_vector_store(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Vector store not found for this doc_id")

    embeddings = get_embeddings()
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )

