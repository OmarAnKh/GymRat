from sentence_transformers import SentenceTransformer
from domain import IEmbeddingService

class SentenceTransformerEmbeddingService(IEmbeddingService):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str):
        return self.model.encode(text)
