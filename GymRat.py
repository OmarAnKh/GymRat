from infrastructure import SentenceTransformerEmbeddingService
from infrastructure import ChromaDBRetriever
from infrastructure import HuggingFaceGenerator, MistralGenerator
from application import load_and_prepare_dataset
from application import RAGChatbot
if __name__ == "__main__":
    embedder = SentenceTransformerEmbeddingService()
    retriever = ChromaDBRetriever()
    generator = MistralGenerator()

    dataset = load_and_prepare_dataset()

    dataset = dataset.map(lambda x: {
        **x,
        'instruction_embedding': embedder.embed(x['instruction'])
    })

    retriever.add_documents(
        embeddings=[x['instruction_embedding'] for x in dataset],
        metadatas=[{'instruction': x['instruction'], 'response': x['response']} for x in dataset],
        ids=[str(i) for i in range(len(dataset))]
    )

    chatbot = RAGChatbot(retriever, generator, embedder)

    question = "Is it possible to build muscle while losing weight?"
    print(chatbot.chat(question))
