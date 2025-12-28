"""
Test script để kiểm tra Context RAG hoạt động
"""

import os
from context_rag import ContextRAG

def test_rag():
    # Khởi tạo Context RAG
    print("=" * 80)
    print("TESTING CONTEXT RAG")
    print("=" * 80)
    
    gemini_key = os.environ.get("GOOGLE_API_KEY")
    if not gemini_key:
        print("ERROR: GOOGLE_API_KEY not set!")
        return
    
    # Test trên CPU vì GPU đang được backend sử dụng
    import torch
    device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
    
    rag = ContextRAG(
        chroma_dir="chroma_db",
        collection_name="vn_traffic_law",
        gemini_api_key=gemini_key,
        gemini_model="gemini-2.5-flash",
        use_rerank=True
    )
    
    print("\n" + "=" * 80)
    print("TEST 1: Dense search (không rerank)")
    print("=" * 80)
    
    question = "Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?"
    print(f"\nQuestion: {question}")
    
    # Test dense search
    results = rag.dense_search(question, top_k=10)
    print(f"\nFound {len(results)} results")
    
    for i, result in enumerate(results[:5], 1):
        print(f"\n--- Result {i} ---")
        print(f"Keys: {result.keys()}")
        dist = result.get('distance', result.get('score', 'N/A'))
        print(f"Distance/Score: {dist}")
        print(f"Content: {result['content'][:200]}...")
        if 'metadata' in result and result['metadata']:
            print(f"Metadata: {result['metadata']}")

    
    print("\n" + "=" * 80)
    print("TEST 2: Rerank results")
    print("=" * 80)
    
    # Test rerank
    reranked = rag.rerank_vi(question, results, top_n=5)
    print(f"\nReranked to {len(reranked)} results")
    
    for i, result in enumerate(reranked, 1):
        print(f"\n--- Reranked {i} (score: {result.get('rerank_score', 0):.4f}) ---")
        print(f"Content: {result['content'][:200]}...")
        if 'clause_number' in result.get('metadata', {}):
            print(f"Clause: {result['metadata']['clause_number']}")
    
    print("\n" + "=" * 80)
    print("TEST 3: Build context")
    print("=" * 80)
    
    context = rag.build_context_from_hits(reranked)
    print(f"\nContext length: {len(context)} chars")
    print(f"Context preview:\n{context[:500]}...")
    
    print("\n" + "=" * 80)
    print("TEST 4: Full answer_question")
    print("=" * 80)
    
    result = rag.answer_question(
        question=question,
        top_k_dense=30,
        top_n_final=10,
        max_tokens=2048
    )
    
    print(f"\nQuestion: {result['question']}")
    print(f"Normalized query: {result['normalized_query']}")
    print(f"Number of hits: {len(result['hits'])}")
    print(f"\nAnswer:\n{result['answer']}")
    
    print("\n" + "=" * 80)
    print("TEST 5: Test với câu hỏi khác")
    print("=" * 80)
    
    question2 = "Không đội mũ bảo hiểm khi lái xe máy bị xử phạt như thế nào?"
    result2 = rag.answer_question(
        question=question2,
        top_k_dense=30,
        top_n_final=10,
        max_tokens=2048
    )
    
    print(f"\nQuestion: {question2}")
    print(f"\nAnswer:\n{result2['answer']}")
    
    print("\n" + "=" * 80)
    print("TEST 6: Test không dùng rerank")
    print("=" * 80)
    
    # Test không rerank
    rag_no_rerank = ContextRAG(
        chroma_dir="chroma_db",
        collection_name="vn_traffic_law",
        gemini_api_key=gemini_key,
        gemini_model="gemini-2.5-flash",
        use_rerank=False
    )
    
    result3 = rag_no_rerank.answer_question(
        question=question,
        top_k_dense=10,
        top_n_final=10,
        max_tokens=2048
    )
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer (no rerank):\n{result3['answer']}")

if __name__ == "__main__":
    test_rag()
