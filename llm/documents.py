import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime


embed_model = SentenceTransformer('all-MiniLM-L6-v2')

with open("neo4j_query_table_data.json", "r") as f:
    neo4j_data = json.load(f)

raw_chunks = []
def format_action_ts(ts_dict):
    if not isinstance(ts_dict, dict):
        return "N/A"
    try:
        dt = datetime(
            ts_dict["year"],
            ts_dict["month"],
            ts_dict["day"],
            ts_dict["hour"],
            ts_dict["minute"],
            ts_dict["second"]
        )
        print("Yes")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as e:
        return "N/A"
    
for record in neo4j_data:
    try:
        text = f"""
            Tenant: {record.get('tenant', 'N/A')}
            User: {record.get('user', 'N/A')}
            Action Timestamp: {format_action_ts(record.get('action_ts', 'N/A'))}
            Action Type : {record.get('action_type', 'N/A')}
            Message: {record.get('message', 'N/A')}
            Execution Start: {record.get('exec_start', 'N/A')}
            Duration: {record.get('exec_duration', 'N/A')}
            Status: {record.get('exec_status', 'N/A')}
            Run Start: {format_action_ts(record.get('run_start', 'N/A'))}
            Run End: {format_action_ts(record.get('run_end', 'N/A'))}
            Avg CPU: {record.get('run_avg_cpu', 'N/A')}
            Instance: {record.get('instance_name', 'N/A')} (ID: {record.get('instance_id', 'N/A')})
            Instance Type: {record.get('instance_type', 'N/A')}
            Instance Plan: {record.get('instance_plan', 'N/A')}
        """
        chunk = text.strip()
        raw_chunks.append(chunk)
    except Exception as e:
        print(f"⚠️ Skipped record due to error: {e}")

with open("data.txt", "r", encoding="utf-8") as f:
    summary_text = f.read()

def chunk_text(text, chunk_size=800):
    sentences = text.split("\n")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + "\n"
        else:
            chunks.append(current.strip())
            current = s + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

summary_chunks = chunk_text(summary_text)
summary_chunks = [c for c in summary_chunks if len(c.strip()) > 10]

def build_faiss(chunks, index_file):
    clean_chunks = [c for c in chunks if isinstance(c, str) and len(c.strip()) > 10]
    embeddings = embed_model.encode(clean_chunks, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)

build_faiss(raw_chunks, "raw_tenant_logs.index")
build_faiss(summary_chunks, "summary_logs.index")

with open("raw_chunks.json", "w") as f: json.dump(raw_chunks, f)
with open("summary_chunks.json", "w") as f: json.dump(summary_chunks, f)

print("Saved")
