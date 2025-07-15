import faiss
import json
import os
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_CHAT")
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324:free"

RAW_INDEX_FILE = "raw_tenant_logs.index"
SUMMARY_INDEX_FILE = "summary_logs.index"
RAW_CHUNKS_FILE = "raw_chunks.json"
SUMMARY_CHUNKS_FILE = "summary_chunks.json"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

raw_index = faiss.read_index(RAW_INDEX_FILE)
summary_index = faiss.read_index(SUMMARY_INDEX_FILE)

with open(RAW_CHUNKS_FILE, "r") as f:
    raw_chunks = json.load(f)

with open(SUMMARY_CHUNKS_FILE, "r") as f:
    summary_chunks = json.load(f)

def retrieve_top_k(query: str, index, chunks, top_k=3):
    query_emb = embed_model.encode([query])
    _, I = index.search(query_emb, top_k)
    return [chunks[i] for i in I[0]]

def retrieve_hybrid(query: str, top_k=3):
    raw_results = retrieve_top_k(query, raw_index, raw_chunks, top_k)
    summary_results = retrieve_top_k(query, summary_index, summary_chunks, top_k)
    return raw_results + summary_results

def query_deepseek(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content']

def normal_chat(user_query: str) -> str:
    return query_deepseek(user_query)

def rag_summary_only(user_query: str) -> str:
    context = "\n\n".join(retrieve_top_k(user_query, raw_index, raw_chunks))
    prompt = f"""
You are analyzing summarized tenant insights.

Context:
{context}

Question: {user_query}
Answer concisely.
"""
    return query_deepseek(prompt)

def rag_hybrid(user_query: str) -> str:
    context = "\n\n".join(retrieve_hybrid(user_query))
    prompt = f"""
You are analyzing tenant activity logs. This includes raw logs and preprocessed summaries.

Context:
{context}

Question: {user_query}
"""
    return query_deepseek(prompt)

app = Flask(__name__)

@app.route("/rag", methods=["POST"])
def rag_endpoint():
    data = request.json
    q = data.get("query")
    mode = data.get("mode", "hybrid").lower()
    print(f"➡️ Query: {q} | Mode: {mode}")
    try:
        if mode == "normal":
            answer = normal_chat(q)
        elif mode == "summary":
            answer = rag_summary_only(q)
        else:
            answer = rag_hybrid(q)

        print("✅ Answer:", answer[:200])
        return jsonify({"query": q, "mode": mode, "answer": answer})

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ ERROR:", tb)
        return jsonify({"query": q, "mode": mode, "error": str(e), "traceback": tb}), 500
    
if __name__ == "__main__":
    app.run(port=5000, debug=True)
