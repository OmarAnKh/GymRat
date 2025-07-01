from domain.interfaces import IEmbeddingService, IRetriever, IGenerator
from application.prompt_builder import format_prompt

class RAGChatbot:
    def __init__(self, retriever: IRetriever, generator: IGenerator, embedder: IEmbeddingService):
        self.retriever = retriever
        self.generator = generator
        self.embedder = embedder

    def chat(self, query: str, k: int = 2) -> str:
        embedding = self.embedder.embed(query)
        results = self.retriever.search(embedding, k)
        prompt = format_prompt(query, results)
        return self.generator.generate(prompt)
