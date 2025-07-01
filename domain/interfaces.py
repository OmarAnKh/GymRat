from abc import ABC, abstractmethod

class IEmbeddingService(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]: pass

class IRetriever(ABC):
    @abstractmethod
    def search(self, query: str, k: int) -> dict: pass

class IGenerator(ABC):
    @abstractmethod
    def generate(self, formatted_prompt: str) -> str: pass
