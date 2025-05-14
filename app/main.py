from fastapi import FastAPI
from app.models.query import Query
from app.retriever import retrieve_documents
from app.generator import generate_answer

app = FastAPI()

@app.post("/chat/")
async def chat(query: Query):
    relevant_docs = retrieve_documents(query.question)
    answer = generate_answer(query.question, relevant_docs)
    return {"answer": answer}
