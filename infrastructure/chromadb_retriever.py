import chromadb
from domain import IRetriever

class ChromaDBRetriever(IRetriever):
    def __init__(self, collection_name: str = "gym-exercise"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, embeddings, metadatas, ids):
        self.collection.add(embeddings=embeddings, metadatas=metadatas, ids=ids)

    def search(self, query_embedding, k: int = 3):
        return self.collection.query(query_embeddings=query_embedding, n_results=k)
