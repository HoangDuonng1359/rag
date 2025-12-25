"""
Traditional RAG - Embeddings Module
Quản lý embedding model và generate embeddings cho documents
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from tqdm import tqdm


class TraditionalRAGEmbeddings:
    """
    Embeddings manager cho Traditional RAG
    
    Sử dụng SentenceTransformers để generate embeddings cho:
    - Document chunks khi indexing
    - User queries khi searching
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: Tên model từ sentence-transformers
            device: 'cuda', 'cpu', hoặc None (auto-detect)
        """
        self.model_name = model_name
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded successfully")
        print(f"  Device: {self.model.device}")
        print(f"  Embedding dimension: {self.embedding_dim}")
    
    def encode_text(
        self,
        text: str,
        normalize: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode một text string thành embedding vector
        
        Args:
            text: Text cần encode
            normalize: Normalize vector về unit length
            convert_to_numpy: Convert sang numpy array
            
        Returns:
            Embedding vector (numpy array hoặc tensor)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=convert_to_numpy,
            show_progress_bar=False
        )
        return embedding
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
        convert_to_numpy: bool = True
    ) -> np.ndarray:
        """
        Encode một batch texts thành embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size cho encoding
            normalize: Normalize vectors
            show_progress: Hiển thị progress bar
            convert_to_numpy: Convert sang numpy
            
        Returns:
            Matrix of embeddings (shape: [len(texts), embedding_dim])
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=convert_to_numpy
        )
        return embeddings
    
    def encode_documents(
        self,
        documents: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Encode documents với progress tracking chi tiết
        
        Args:
            documents: List of document strings
            batch_size: Batch size
            show_progress: Hiển thị progress
            
        Returns:
            List of embedding vectors
        """
        total = len(documents)
        all_embeddings = []
        
        if show_progress:
            print(f"\nEncoding {total:,} documents...")
            iterator = tqdm(range(0, total, batch_size), desc="Encoding")
        else:
            iterator = range(0, total, batch_size)
        
        for i in iterator:
            batch = documents[i:i + batch_size]
            
            # Encode batch
            batch_embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Add to results
            all_embeddings.extend(batch_embeddings)
        
        if show_progress:
            print(f"✓ Encoded {total:,} documents")
        
        return all_embeddings
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Tính similarity giữa query và documents
        
        Args:
            query_embedding: Query vector (shape: [embedding_dim])
            document_embeddings: Document vectors (shape: [n_docs, embedding_dim])
            metric: 'cosine' hoặc 'dot'
            
        Returns:
            Similarity scores (shape: [n_docs])
        """
        if metric == "cosine":
            # Cosine similarity (assuming normalized vectors)
            similarities = np.dot(document_embeddings, query_embedding)
        elif metric == "dot":
            # Dot product
            similarities = np.dot(document_embeddings, query_embedding)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return similarities
    
    def get_top_k_similar(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
        top_k: int = 5
    ) -> tuple:
        """
        Tìm top-k documents tương tự nhất với query
        
        Args:
            query_embedding: Query vector
            document_embeddings: Document vectors matrix
            top_k: Số lượng kết quả
            
        Returns:
            (indices, similarities) - indices của top-k docs và similarity scores
        """
        # Compute similarities
        similarities = self.compute_similarity(query_embedding, document_embeddings)
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        top_k_similarities = similarities[top_k_indices]
        
        return top_k_indices, top_k_similarities
    
    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode user query (alias cho encode_text với sensible defaults)
        
        Args:
            query: Query string
            normalize: Normalize vector
            
        Returns:
            Query embedding vector
        """
        return self.encode_text(query, normalize=normalize)
    
    def get_model_info(self) -> dict:
        """
        Lấy thông tin về model
        
        Returns:
            Dict với model info
        """
        return {
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim,
            'device': str(self.model.device),
            'max_seq_length': self.model.max_seq_length
        }


# Example usage và tests
if __name__ == "__main__":
    # Initialize embeddings
    print("=== Initializing Embeddings ===")
    embeddings = TraditionalRAGEmbeddings()
    
    # Model info
    print("\n=== Model Info ===")
    info = embeddings.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test encoding single text
    print("\n=== Test Single Text Encoding ===")
    text = "Cách mạng tháng Tám năm 1945 thành công"
    embedding = embeddings.encode_text(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"First 5 values: {embedding[:5]}")
    
    # Test batch encoding
    print("\n=== Test Batch Encoding ===")
    texts = [
        "Hồ Chí Minh lãnh đạo cách mạng",
        "Chiến dịch Điện Biên Phủ",
        "Việt Nam dân chủ cộng hòa",
        "Kháng chiến chống Pháp",
        "Mặt trận Việt Minh"
    ]
    
    batch_embeddings = embeddings.encode_batch(texts, batch_size=2, show_progress=True)
    print(f"Encoded {len(texts)} texts")
    print(f"Embeddings shape: {batch_embeddings.shape}")
    
    # Test similarity computation
    print("\n=== Test Similarity Computation ===")
    query = "người lãnh đạo phong trào cách mạng"
    query_embedding = embeddings.encode_query(query)
    
    similarities = embeddings.compute_similarity(query_embedding, batch_embeddings)
    
    print(f"Query: {query}")
    print("\nSimilarities:")
    for i, (text, sim) in enumerate(zip(texts, similarities)):
        print(f"{i+1}. {text}")
        print(f"   Similarity: {sim:.4f}")
    
    # Test top-k retrieval
    print("\n=== Test Top-K Retrieval ===")
    top_k = 3
    indices, scores = embeddings.get_top_k_similar(
        query_embedding, 
        batch_embeddings, 
        top_k=top_k
    )
    
    print(f"Top-{top_k} results for: '{query}'")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"{rank}. {texts[idx]}")
        print(f"   Score: {score:.4f}")
    
    # Test with longer documents
    print("\n=== Test with Document Chunks ===")
    import json
    
    # Load sample chunks
    with open('./data/chunk/chapter10_chunk.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Take first 10 chunks
    sample_chunks = data['chunks'][:10]
    chunk_texts = [chunk['content'] for chunk in sample_chunks]
    
    print(f"Encoding {len(chunk_texts)} sample chunks...")
    chunk_embeddings = embeddings.encode_documents(
        chunk_texts,
        batch_size=5,
        show_progress=True
    )
    
    # Search in chunks
    query = "Cách mạng tháng Tám thành công do đâu?"
    query_emb = embeddings.encode_query(query)
    
    chunk_embeddings_array = np.array(chunk_embeddings)
    indices, scores = embeddings.get_top_k_similar(
        query_emb,
        chunk_embeddings_array,
        top_k=3
    )
    
    print(f"\nQuery: {query}")
    print(f"Top-3 chunks:")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"\n{rank}. Chunk {sample_chunks[idx]['id']} (score: {score:.4f})")
        print(f"   Pages: {sample_chunks[idx]['metadata']['page_start']}-{sample_chunks[idx]['metadata']['page_end']}")
        print(f"   Content: {chunk_texts[idx][:100]}...")
