"""
Traditional RAG - ChromaDB Vector Store
Quản lý ChromaDB collection cho Traditional RAG system
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import numpy as np
from traditional_rag_embeddings import TraditionalRAGEmbeddings


class TraditionalRAGVectorStore:
    """
    ChromaDB Vector Store cho Traditional RAG
    
    Lưu trữ document chunks với embeddings và metadata
    """
    
    def __init__(
        self,
        collection_name: str = "vietnamese_history_chunks",
        persist_directory: str = "./chroma_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Tên collection trong ChromaDB
            persist_directory: Thư mục lưu trữ database
            model_name: Model để generate embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.model_name = model_name
        
        # Initialize ChromaDB client với persistent storage
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model using TraditionalRAGEmbeddings
        print(f"Initializing embedding model...")
        self.embeddings = TraditionalRAGEmbeddings(model_name=model_name)
        self.embedding_dim = self.embeddings.embedding_dim
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self):
        """Get existing collection hoặc tạo mới"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            count = collection.count()
            print(f"Found existing collection '{self.collection_name}' with {count:,} chunks")
            return collection
        except Exception:
            # Create new collection with cosine similarity
            print(f"Creating new collection '{self.collection_name}'")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "Vietnamese History Document Chunks",
                    "hnsw:space": "cosine"  # Sử dụng cosine similarity
                }
            )
            return collection
    
    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Add document chunks vào vector store
        
        Args:
            chunks: List of chunk dicts với keys: id, content, metadata
            batch_size: Số chunks xử lý cùng lúc
            show_progress: Hiển thị progress
            
        Returns:
            Stats dict với success/failed counts
        """
        total = len(chunks)
        success = 0
        failed = 0
        
        if show_progress:
            print(f"\nAdding {total:,} chunks to ChromaDB...")
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                # Extract data từ batch
                ids = [chunk['id'] for chunk in batch]
                documents = [chunk['content'] for chunk in batch]
                
                # Filter out None values from metadata (ChromaDB không chấp nhận None)
                metadatas = []
                for chunk in batch:
                    clean_metadata = {
                        k: v for k, v in chunk['metadata'].items() 
                        if v is not None
                    }
                    metadatas.append(clean_metadata)
                
                # Generate embeddings using TraditionalRAGEmbeddings
                embeddings_list = self.embeddings.encode_batch(
                    documents,
                    batch_size=len(batch),
                    show_progress=False,
                    normalize=True
                )
                
                # Add to collection
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings_list.tolist(),
                    metadatas=metadatas
                )
                
                success += len(batch)
                
                if show_progress and (i + batch_size) % 100 == 0:
                    print(f"  Processed {min(i + batch_size, total):,}/{total:,} chunks...")
                    
            except Exception as e:
                print(f"Error adding batch {i}-{i+batch_size}: {e}")
                failed += len(batch)
        
        if show_progress:
            print(f"\n✓ Added {success:,} chunks successfully")
            if failed > 0:
                print(f"✗ Failed: {failed} chunks")
        
        return {
            'total': total,
            'success': success,
            'failed': failed
        }
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search cho chunks tương tự query
        
        Args:
            query: Câu hỏi/query string
            top_k: Số lượng chunks trả về
            filter_metadata: Filter theo metadata (e.g., {'chapter': 'I'})
            
        Returns:
            List of dicts với keys: id, content, metadata, distance, similarity
        """
        # Generate query embedding using TraditionalRAGEmbeddings
        query_embedding = self.embeddings.encode_query(query)
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            chunk_id = results['ids'][0][i]
            content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # ChromaDB với cosine distance trả về: distance = 1 - cosine_similarity
            # Nên similarity = 1 - distance
            similarity = 1.0 - distance
            
            formatted_results.append({
                'id': chunk_id,
                'content': content,
                'metadata': metadata,
                'distance': distance,
                'similarity': similarity
            })
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy một chunk cụ thể theo ID
        
        Args:
            chunk_id: ID của chunk
            
        Returns:
            Chunk dict hoặc None
        """
        try:
            result = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )
            
            if len(result['ids']) > 0:
                return {
                    'id': result['ids'][0],
                    'content': result['documents'][0],
                    'metadata': result['metadatas'][0],
                    'embedding': result['embeddings'][0]
                }
            return None
            
        except Exception as e:
            print(f"Error getting chunk {chunk_id}: {e}")
            return None
    
    def get_chunks_by_page(self, page_number: int) -> List[Dict[str, Any]]:
        """
        Lấy tất cả chunks từ một page cụ thể
        
        Args:
            page_number: Số trang
            
        Returns:
            List of chunks
        """
        results = self.collection.get(
            where={
                "$and": [
                    {"page_start": {"$lte": page_number}},
                    {"page_end": {"$gte": page_number}}
                ]
            },
            include=['documents', 'metadatas']
        )
        
        chunks = []
        for i in range(len(results['ids'])):
            chunks.append({
                'id': results['ids'][i],
                'content': results['documents'][i],
                'metadata': results['metadatas'][i]
            })
        
        return chunks
    
    def count_chunks(self) -> int:
        """Đếm tổng số chunks trong collection"""
        return self.collection.count()
    
    def delete_all(self) -> bool:
        """
        Xóa toàn bộ collection và tạo lại mới
        
        Returns:
            True nếu thành công
        """
        try:
            print(f"Deleting collection '{self.collection_name}'...")
            self.client.delete_collection(name=self.collection_name)
            
            print("Creating new empty collection...")
            self.collection = self._get_or_create_collection()
            
            print("✓ Collection reset successfully")
            return True
            
        except Exception as e:
            print(f"Error resetting collection: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Lấy thống kê về collection
        
        Returns:
            Dict với stats
        """
        count = self.collection.count()
        
        # Get sample để check metadata
        sample = self.collection.peek(limit=1)
        
        stats = {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'persist_directory': self.persist_directory
        }
        
        if count > 0 and len(sample['metadatas']) > 0:
            # Check metadata fields
            sample_metadata = sample['metadatas'][0]
            stats['metadata_fields'] = list(sample_metadata.keys())
        
        return stats


# Example usage
if __name__ == "__main__":
    import json
    
    # Initialize vector store
    print("=== Initializing ChromaDB Vector Store ===")
    vectorstore = TraditionalRAGVectorStore(
        collection_name="vietnamese_history_chunks",
        persist_directory="./chroma_db"
    )
    
    # Load chunks từ file
    print("\n=== Loading Chunks from JSON ===")
    with open('./data/chunk/chapter10_chunk.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = data['chunks']
    print(f"Loaded {len(chunks):,} chunks")
    print(f"Config: {data['config']}")
    
    # Add chunks to vectorstore (comment out if already added)
    # stats = vectorstore.add_chunks(chunks, batch_size=32)
    # print(f"\n=== Add Stats ===")
    # print(f"Total: {stats['total']}")
    # print(f"Success: {stats['success']}")
    # print(f"Failed: {stats['failed']}")
    
    # Get stats
    print("\n=== Vector Store Stats ===")
    stats = vectorstore.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Test search
    print("\n=== Testing Semantic Search ===")
    test_queries = [
        "Quy định về nồng độ cồn khi lái xe",
        "Mức phạt vi phạm tốc độ trên đường cao tốc",
        "Các biển báo giao thông quan trọng"
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        results = vectorstore.search_similar(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Chunk ID: {result['id']}")
            print(f"   Similarity: {result['similarity']:.4f}")
            print(f"   Pages: {result['metadata']['page_start']}-{result['metadata']['page_end']}")
            print(f"   Content: {result['content'][:150]}...")
