"""
Traditional RAG - Retrieval System
High-level interface cho retrieval với preprocessing và postprocessing
"""

from typing import List, Dict, Any, Optional
from traditional_rag_vectorstore import TraditionalRAGVectorStore
import time


class TraditionalRAGRetriever:
    """
    High-level retrieval system cho Traditional RAG
    
    Wrapper xung quanh VectorStore với additional features:
    - Query preprocessing
    - Result filtering và ranking
    - Metrics tracking
    - Caching (optional)
    """
    
    def __init__(
        self,
        vectorstore: Optional[TraditionalRAGVectorStore] = None,
        collection_name: str = "vietnamese_history_chunks",
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize retriever
        
        Args:
            vectorstore: Existing vectorstore instance (nếu None sẽ tạo mới)
            collection_name: Tên collection
            persist_directory: Directory cho ChromaDB
        """
        if vectorstore is None:
            print("Initializing new vector store...")
            self.vectorstore = TraditionalRAGVectorStore(
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        else:
            self.vectorstore = vectorstore
        
        # Metrics
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0
        
        print(f"✓ Retriever initialized")
        print(f"  Vector store: {self.vectorstore.count_chunks():,} chunks")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks cho query
        
        Args:
            query: User query string
            top_k: Số lượng chunks trả về
            min_similarity: Minimum similarity threshold (filter ra results < threshold)
            filter_metadata: Filter theo metadata
            
        Returns:
            List of retrieved chunks với metadata và scores
        """
        start_time = time.time()
        
        # Preprocess query (có thể thêm logic sau)
        processed_query = self._preprocess_query(query)
        
        # Retrieve từ vector store
        results = self.vectorstore.search_similar(
            processed_query,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        # Filter by similarity threshold
        if min_similarity is not None:
            results = [r for r in results if r['similarity'] >= min_similarity]
        
        # Post-process results (có thể thêm reranking sau)
        processed_results = self._postprocess_results(results)
        
        # Update metrics
        retrieval_time = time.time() - start_time
        self.retrieval_count += 1
        self.total_retrieval_time += retrieval_time
        
        return processed_results
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query trước khi search
        
        Args:
            query: Raw query
            
        Returns:
            Processed query
        """
        # Hiện tại chỉ return as-is
        # Có thể thêm: query expansion, spelling correction, etc.
        return query.strip()
    
    def _postprocess_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Post-process results sau khi retrieve
        
        Args:
            results: Raw results từ vector store
            
        Returns:
            Processed results
        """
        # Hiện tại chỉ return as-is
        # Có thể thêm: reranking, deduplication, etc.
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve với context window (lấy thêm chunks xung quanh)
        
        Args:
            query: User query
            top_k: Số lượng chunks chính
            context_window: Số chunks lấy thêm trước/sau mỗi chunk
            
        Returns:
            Retrieved chunks với optional context chunks
        """
        # Retrieve main chunks
        results = self.retrieve(query, top_k=top_k)
        
        if context_window == 0:
            return results
        
        # TODO: Implement context window expansion
        # Cần có chunk ordering/positioning để lấy adjacent chunks
        # Hiện tại chỉ return main results
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Batch retrieval cho nhiều queries
        
        Args:
            queries: List of query strings
            top_k: Số chunks cho mỗi query
            
        Returns:
            List of results lists
        """
        all_results = []
        
        for query in queries:
            results = self.retrieve(query, top_k=top_k)
            all_results.append(results)
        
        return all_results
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Lấy retrieval metrics
        
        Returns:
            Dict với metrics
        """
        avg_time = (
            self.total_retrieval_time / self.retrieval_count 
            if self.retrieval_count > 0 
            else 0.0
        )
        
        return {
            'total_retrievals': self.retrieval_count,
            'total_time_seconds': self.total_retrieval_time,
            'average_time_seconds': avg_time,
            'chunks_in_index': self.vectorstore.count_chunks()
        }
    
    def reset_metrics(self):
        """Reset metrics counters"""
        self.retrieval_count = 0
        self.total_retrieval_time = 0.0
        print("✓ Metrics reset")


# Example usage và tests
if __name__ == "__main__":
    print("=== Initializing Retriever ===")
    retriever = TraditionalRAGRetriever()
    
    # Test single retrieval
    print("\n=== Test Single Retrieval ===")
    query = "Cách mạng tháng Tám năm 1945 thành công do đâu?"
    
    print(f"Query: {query}")
    print("="*80)
    
    results = retriever.retrieve(query, top_k=5)
    
    print(f"\nRetrieved {len(results)} chunks:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Similarity: {result['similarity']:.4f}")
        print(f"   Chunk ID: {result['id']}")
        print(f"   Pages: {result['metadata']['page_start']}-{result['metadata']['page_end']}")
        print(f"   Content: {result['content'][:120]}...")
    
    # Test với similarity threshold
    print("\n=== Test with Similarity Threshold ===")
    results_filtered = retriever.retrieve(
        query,
        top_k=10,
        min_similarity=0.6
    )
    
    print(f"\nRetrieved {len(results_filtered)} chunks with similarity >= 0.6:")
    for i, result in enumerate(results_filtered, 1):
        print(f"{i}. {result['id']}: {result['similarity']:.4f}")
    
    # Test multiple queries
    print("\n=== Test Multiple Queries ===")
    test_queries = [
        "Ai là người lãnh đạo cách mạng?",
        "Các chiến dịch quân sự quan trọng",
        "Chính phủ Việt Nam dân chủ cộng hòa"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=3)
        
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['id']} - similarity: {r['similarity']:.4f}")
    
    # Test batch retrieval
    print("\n=== Test Batch Retrieval ===")
    batch_results = retriever.batch_retrieve(test_queries, top_k=2)
    
    print(f"Processed {len(batch_results)} queries")
    for i, (query, results) in enumerate(zip(test_queries, batch_results), 1):
        print(f"\n{i}. {query}")
        print(f"   Retrieved {len(results)} chunks")
    
    # Show metrics
    print("\n=== Retrieval Metrics ===")
    metrics = retriever.get_metrics()
    for key, value in metrics.items():
        if 'time' in key:
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Test performance
    print("\n=== Performance Test ===")
    retriever.reset_metrics()
    
    import time
    start = time.time()
    
    for _ in range(10):
        retriever.retrieve("test query", top_k=5)
    
    elapsed = time.time() - start
    
    print(f"10 retrievals in {elapsed:.2f}s")
    print(f"Average: {elapsed/10:.4f}s per retrieval")
    
    metrics = retriever.get_metrics()
    print(f"Metrics average: {metrics['average_time_seconds']:.4f}s")
