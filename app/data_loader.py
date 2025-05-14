import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class KnowledgeBase:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_file='faiss_index.pkl'):
        self.model = SentenceTransformer(model_name)
        self.index_file = index_file
        self.documents = []
        self.embeddings = None
        self.index = None

    def load_documents(self, filepath='documents/knowledge_base.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.documents = f.readlines()
        self.embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        with open(self.index_file, 'wb') as f:
            pickle.dump((self.index, self.documents), f)

    def load_index(self):
        with open(self.index_file, 'rb') as f:
            self.index, self.documents = pickle.load(f)

    def get_relevant_docs(self, query, k=3):
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        return [self.documents[i] for i in indices[0]]
