import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from context_rag import ContextRAG

gemini_key = os.environ.get("GOOGLE_API_KEY")
rag = ContextRAG('chroma_db', 'vn_traffic_law', gemini_api_key=gemini_key, use_rerank=True)

q = 'Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?'
results = rag.dense_search(q, top_k=10)
reranked = rag.rerank_vi(q, results, top_n=5)

print('Rerank scores:')
for i, r in enumerate(reranked, 1):
    score = r.get("rerank_score", r.get("score_rerank", 0))
    print(f'{i}. Score: {score:.4f} - {r["content"][:80]}...')
