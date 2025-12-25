"""
Graph RAG - Vector Embeddings for Entities
Generate và store semantic embeddings cho entities để enable semantic search
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm
import json


class EntityEmbeddings:
    """Class để manage entity embeddings cho semantic search"""
    
    def __init__(
        self, 
        graph: Neo4jGraph,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    ):
        """
        Initialize với Neo4j graph và sentence transformer model
        
        Args:
            graph: Neo4jGraph instance
            model_name: HuggingFace model name cho embeddings
                       Default: multilingual model support tiếng Việt
        """
        self.graph = graph
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded! Embedding dimension: {self.embedding_dim}")
    
    def _create_entity_text(self, entity: Dict) -> str:
        """
        Tạo text representation của entity để embed
        
        Args:
            entity: Dict với keys: name, type, description
            
        Returns:
            String combines name, type, và description
        """
        name = entity.get('name', '')
        entity_type = entity.get('type', '')
        description = entity.get('description', '')
        
        # Format: "Tên (Loại): Mô tả"
        if description:
            return f"{name} ({entity_type}): {description}"
        else:
            return f"{name} ({entity_type})"
    
    def get_all_entities(self) -> List[Dict]:
        """
        Lấy tất cả entities từ Neo4j
        
        Returns:
            List of entity dicts với name, type, description, page info
        """
        query = """
        MATCH (n)
        WHERE n.name IS NOT NULL
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description,
               id(n) as node_id,
               n.first_seen_page as first_seen_page,
               n.first_seen_chapter as first_seen_chapter
        ORDER BY n.name
        """
        
        results = self.graph.query(query)
        print(f"Found {len(results)} entities in graph")
        return results
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings cho list of texts
        
        Args:
            texts: List of strings to embed
            batch_size: Batch size cho encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings
    
    def add_embedding_to_entity(
        self, 
        entity_name: str,
        embedding: List[float]
    ) -> bool:
        """
        Store embedding vào Neo4j entity
        
        Args:
            entity_name: Tên entity
            embedding: Vector embedding
            
        Returns:
            True if success
        """
        # Neo4j store list of floats as property
        query = """
        MATCH (n {name: $name})
        SET n.embedding = $embedding
        RETURN n.name as name
        """
        
        try:
            result = self.graph.query(query, {
                'name': entity_name,
                'embedding': embedding
            })
            return len(result) > 0
        except Exception as e:
            print(f"Error storing embedding for {entity_name}: {e}")
            return False
    
    def generate_and_store_all_embeddings(
        self,
        batch_size: int = 32
    ) -> Dict[str, int]:
        """
        Generate embeddings cho tất cả entities và store vào Neo4j
        
        Args:
            batch_size: Batch size cho encoding
            
        Returns:
            Dict với statistics
        """
        print("Starting embedding generation...")
        
        # Get all entities
        entities = self.get_all_entities()
        
        if not entities:
            print("No entities found in graph!")
            return {"success": 0, "failed": 0, "total": 0}
        
        # Create texts for embedding
        print("Creating entity texts...")
        entity_texts = [self._create_entity_text(e) for e in entities]
        
        # Generate embeddings
        print(f"Generating embeddings for {len(entity_texts)} entities...")
        embeddings = self.generate_embeddings_batch(
            entity_texts, 
            batch_size=batch_size,
            show_progress=True
        )
        
        # Store embeddings back to Neo4j
        print("Storing embeddings to Neo4j...")
        success_count = 0
        failed_count = 0
        
        for entity, embedding in tqdm(zip(entities, embeddings), total=len(entities)):
            # Convert numpy array to list for Neo4j
            embedding_list = embedding.tolist()
            
            if self.add_embedding_to_entity(entity['name'], embedding_list):
                success_count += 1
            else:
                failed_count += 1
        
        stats = {
            "success": success_count,
            "failed": failed_count,
            "total": len(entities),
            "embedding_dim": self.embedding_dim
        }
        
        print(f"\n{'='*50}")
        print(f"EMBEDDINGS GENERATED!")
        print(f"{'='*50}")
        print(f"Success: {success_count}/{len(entities)}")
        print(f"Failed: {failed_count}")
        print(f"Embedding dimension: {self.embedding_dim}")
        
        return stats
    
    def get_entities_with_embeddings(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Lấy entities có embeddings từ Neo4j
        
        Args:
            limit: Giới hạn số lượng, None = all
            
        Returns:
            List of entities với embeddings
        """
        query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description,
               n.embedding as embedding
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        results = self.graph.query(query)
        return results
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        filter_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Semantic search cho entities dựa vào query string
        
        Args:
            query: Search query
            top_k: Số lượng kết quả
            filter_types: Chỉ search trong entity types này (e.g. ['PERSON', 'LOCATION'])
            
        Returns:
            List of entities với similarity scores
        """
        # Generate embedding cho query
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        
        # Get all entities with embeddings
        entities = self.get_entities_with_embeddings()
        
        if not entities:
            print("No entities with embeddings found!")
            return []
        
        # Filter by type if specified
        if filter_types:
            entities = [e for e in entities if e['type'] in filter_types]
        
        # Calculate similarities
        results = []
        for entity in entities:
            entity_embedding = np.array(entity['embedding'])
            
            # Cosine similarity
            similarity = self._cosine_similarity(query_embedding, entity_embedding)
            
            results.append({
                'name': entity['name'],
                'type': entity['type'],
                'description': entity['description'],
                'similarity': float(similarity),
                'first_seen_page': entity.get('first_seen_page'),
                'first_seen_chapter': entity.get('first_seen_chapter')
            })
        
        # Sort by similarity và return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def find_similar_entities(
        self,
        entity_name: str,
        top_k: int = 10,
        exclude_self: bool = True
    ) -> List[Dict]:
        """
        Tìm entities tương tự với một entity cho trước
        
        Args:
            entity_name: Tên entity gốc
            top_k: Số lượng similar entities
            exclude_self: Loại bỏ chính entity đó khỏi results
            
        Returns:
            List of similar entities với similarity scores
        """
        # Get target entity embedding
        query = """
        MATCH (n {name: $name})
        WHERE n.embedding IS NOT NULL
        RETURN n.embedding as embedding
        """
        
        result = self.graph.query(query, {'name': entity_name})
        
        if not result or not result[0]['embedding']:
            print(f"Entity '{entity_name}' not found or has no embedding")
            return []
        
        target_embedding = np.array(result[0]['embedding'])
        
        # Get all entities
        entities = self.get_entities_with_embeddings()
        
        # Calculate similarities
        results = []
        for entity in entities:
            if exclude_self and entity['name'] == entity_name:
                continue
            
            entity_embedding = np.array(entity['embedding'])
            similarity = self._cosine_similarity(target_embedding, entity_embedding)
            
            results.append({
                'name': entity['name'],
                'type': entity['type'],
                'description': entity['description'],
                'similarity': float(similarity)
            })
        
        # Sort và return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def check_embedding_coverage(self) -> Dict:
        """
        Kiểm tra coverage của embeddings trong graph
        
        Returns:
            Dict với statistics về embedding coverage
        """
        # Total entities
        total_query = "MATCH (n) RETURN count(n) as count"
        total = self.graph.query(total_query)[0]['count']
        
        # Entities with embeddings
        with_embedding_query = """
        MATCH (n)
        WHERE n.embedding IS NOT NULL
        RETURN count(n) as count
        """
        with_embedding = self.graph.query(with_embedding_query)[0]['count']
        
        # Coverage by type
        type_coverage_query = """
        MATCH (n)
        WITH labels(n)[0] as type, 
             count(n) as total,
             sum(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) as with_embedding
        RETURN type, total, with_embedding,
               toFloat(with_embedding) / total * 100 as coverage_percent
        ORDER BY total DESC
        """
        
        type_coverage = self.graph.query(type_coverage_query)
        
        return {
            'total_entities': total,
            'entities_with_embeddings': with_embedding,
            'coverage_percent': (with_embedding / total * 100) if total > 0 else 0,
            'coverage_by_type': type_coverage
        }
    
    def update_embeddings_for_new_entities(self) -> Dict:
        """
        Update embeddings chỉ cho entities mới (chưa có embedding)
        
        Returns:
            Dict với statistics
        """
        # Get entities without embeddings
        query = """
        MATCH (n)
        WHERE n.embedding IS NULL AND n.name IS NOT NULL
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description
        """
        
        entities = self.graph.query(query)
        
        if not entities:
            print("All entities already have embeddings!")
            return {"new_embeddings": 0, "total": 0}
        
        print(f"Found {len(entities)} entities without embeddings")
        
        # Generate embeddings
        entity_texts = [self._create_entity_text(e) for e in entities]
        embeddings = self.generate_embeddings_batch(entity_texts, show_progress=True)
        
        # Store embeddings
        success_count = 0
        for entity, embedding in zip(entities, embeddings):
            if self.add_embedding_to_entity(entity['name'], embedding.tolist()):
                success_count += 1
        
        print(f"Updated {success_count} new embeddings")
        
        return {"new_embeddings": success_count, "total": len(entities)}


# Example usage
if __name__ == "__main__":
    from langchain_community.graphs import Neo4jGraph
    
    # Setup Neo4j
    NEO4J_URI = "neo4j+s://0c367113.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "gTO1K567hBLzkRdUAhhEb-UqvBjz0i3ckV3M9v_-Nio"
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize embeddings
    embeddings = EntityEmbeddings(graph)
    
    # Generate và store embeddings
    stats = embeddings.generate_and_store_all_embeddings(batch_size=32)
    
    # Test semantic search
    print("\n=== Test Semantic Search ===")
    results = embeddings.semantic_search("lãnh đạo cách mạng", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']} ({r['type']}) - similarity: {r['similarity']:.3f}")
