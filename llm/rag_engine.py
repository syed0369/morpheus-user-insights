import faiss
import json
import os
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_CHAT")
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324:free"

RAW_INDEX_FILE = "llm/raw_tenant_logs.index"
SUMMARY_INDEX_FILE = "llm/summary_logs.index"
RAW_CHUNKS_FILE = "llm/raw_chunks.json"
SUMMARY_CHUNKS_FILE = "llm/summary_chunks.json"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

raw_index = faiss.read_index(RAW_INDEX_FILE)
summary_index = faiss.read_index(SUMMARY_INDEX_FILE)

with open(RAW_CHUNKS_FILE, "r") as f:
    raw_chunks = json.load(f)

with open(SUMMARY_CHUNKS_FILE, "r") as f:
    summary_chunks = json.load(f)

# --- Modified top-k function with distance printing
def retrieve_top_k(query: str, index, chunks, top_k):
    query_emb = embed_model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_emb, top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if dist > 2:
            continue
        chunk = chunks[idx]
        results.append(chunk)
    return results

def retrieve_hybrid(query, top_k1=100, top_k2=25):
    raw_results = retrieve_top_k(query, raw_index, raw_chunks, top_k1)
    summary_results = retrieve_top_k(query, summary_index, summary_chunks, top_k2)
    return raw_results + summary_results

def query_deepseek(prompt):
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message.content

def rag_summary_only(user_query):
    context = "\n\n".join(retrieve_top_k(user_query, summary_index, summary_chunks))
    prompt = f"""
                You are analyzing summarized tenant insights.

                Context:
                {context}

                Question: {user_query}
                Answer concisely.
            """
    return query_deepseek(prompt)

def rag_hybrid(user_query):
    context = "\n\n".join(retrieve_hybrid(user_query))
    prompt = f"""
                You are analyzing tenant activity logs. This includes raw logs and preprocessed summaries.

                Context:
                {context}

                Question: {user_query}
            """
    response = query_deepseek(prompt)
    # print(response)
    return response
