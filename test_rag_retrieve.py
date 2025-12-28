"""
Script test chỉ phần retrieve (không cần Gemini API)
"""

import os
from context_rag import ContextRAG

# Test retrieve only (không cần API key)
rag_system = ContextRAG(
    chroma_dir="chroma_db",
    collection_name="vn_traffic_law",
    gemini_api_key=None,  # Không cần API
    use_rerank=True,
)

# Test query
question = "Mức phạt vượt đèn đỏ đối với xe máy?"
print(f"\n{'='*60}")
print(f"Question: {question}")
print(f"{'='*60}\n")

result = rag_system.rag_retrieve(
    query=question,
    top_k_dense=30,
    top_n_final=10,
)

print(f"Normalized query: {result['normalized_query']}\n")
print(f"Retrieved {len(result['hits'])} documents:\n")

for i, hit in enumerate(result['hits'], 1):
    print(f"\n--- Document {i} ---")
    print(f"Score: {hit.get('score_rerank', hit.get('score_dense', 0)):.4f}")
    meta = hit.get('metadata', {})
    print(f"Source: {meta.get('law_name', 'N/A')}")
    print(f"Article: {meta.get('article', 'N/A')}, Clause: {meta.get('clause', 'N/A')}")
    print(f"Content: {hit['content'][:200]}...")

print(f"\n{'='*60}")
print("CONTEXT:")
print(f"{'='*60}\n")
print(result['context'][:1000])
print("\n... (truncated)")
