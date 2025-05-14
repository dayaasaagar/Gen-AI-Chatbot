import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_answer(query: str, context: list[str]) -> str:
    prompt = f"""
You are a helpful AI assistant. Use the following context to answer the question.

Context:
{''.join(context)}

Question: {query}
Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.5,
    )
    return response['choices'][0]['message']['content']
