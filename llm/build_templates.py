import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load query templates
with open("query_templates.json", "r") as f:
    query_templates = json.load(f)

# Use natural language description ("nl_query") for semantic matching
template_texts = [qt.get("nl_query", qt["cypher"]) for qt in query_templates]

# Encode and normalize for cosine similarity
template_embeddings = embed_model.encode(template_texts, normalize_embeddings=True)
dim = template_embeddings.shape[1]

# Use inner product index for cosine similarity (since vectors are normalized)
index = faiss.IndexFlatIP(dim)
index.add(np.array(template_embeddings))

# Save index and human-readable template list
faiss.write_index(index, "template_embeddings.index")
with open("templates_only.json", "w") as f:
    json.dump(template_texts, f)

print(f"✅ Indexed {len(template_texts)} templates using cosine similarity (dim={dim})")
