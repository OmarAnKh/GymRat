from fastapi import FastAPI
from pydantic import BaseModel
from infrastructure import SentenceTransformerEmbeddingService
from infrastructure import ChromaDBRetriever
from infrastructure import HuggingFaceGenerator,MistralGenerator
from application import load_and_prepare_dataset
from application import RAGChatbot
from fastapi.responses import JSONResponse
import uvicorn


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



class Query(BaseModel):
    question: str

app = FastAPI()


@app.post("/chat/")
async def create_chat(query: Query):
    try:
        chatbot = RAGChatbot(retriever, generator, embedder)
        return JSONResponse(status_code=200, content={"Answer": chatbot.chat(query.question)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"Error": str(e)})

uvicorn.run(app, host="0.0.0.0", port=8000)