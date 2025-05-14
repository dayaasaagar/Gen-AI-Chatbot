from app.data_loader import KnowledgeBase

kb = KnowledgeBase()
try:
    kb.load_index()
except FileNotFoundError:
    kb.load_documents()

def retrieve_documents(query: str, top_k: int = 3):
    return kb.get_relevant_docs(query, k=top_k)
