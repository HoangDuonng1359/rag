"""
Script để regenerate embeddings với context được cải thiện
"""

from langchain_community.graphs import Neo4jGraph
from graph_rag_embeddings import EntityEmbeddings
import os

def main():
    # Setup Neo4j connection
    NEO4J_URI = "neo4j+s://41ab799a.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "xmriUzmvo9dSAyc10u9mpB7nzyQHMZFooKqH5yBP2d4"
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize embeddings với context cải thiện
    print("Initializing embedding model...")
    embeddings = EntityEmbeddings(graph)
    
    # Test xem text được tạo như thế nào với entity mẫu
    print("\n" + "="*60)
    print("TESTING NEW ENTITY TEXT GENERATION")
    print("="*60)
    
    test_entities = ["Vượt đèn đỏ", "Xe máy", "Phạt tiền từ 4-6 triệu đồng"]
    for entity_name in test_entities:
        query = """
        MATCH (n {name: $name})
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description
        """
        result = graph.query(query, {'name': entity_name})
        if result:
            entity = result[0]
            entity_text = embeddings._create_entity_text(entity, include_relationships=True)
            print(f"\n{entity_name}:")
            print(f"  {entity_text}")
    
    # Hỏi user có muốn regenerate không
    print("\n" + "="*60)
    response = input("\nBạn có muốn regenerate tất cả embeddings với context mới? (y/n): ")
    
    if response.lower() == 'y':
        print("\nRegenerating embeddings...")
        stats = embeddings.generate_and_store_all_embeddings(batch_size=32)
        
        print("\n" + "="*60)
        print("REGENERATION COMPLETE")
        print("="*60)
        print(f"Total entities: {stats['total']}")
        print(f"Success: {stats['success']}")
        print(f"Failed: {stats['failed']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        
        # Test semantic search với câu hỏi mẫu
        print("\n" + "="*60)
        print("TESTING SEMANTIC SEARCH")
        print("="*60)
        
        question = "Mức phạt vi phạm vượt đèn đỏ đối với xe máy là bao nhiêu?"
        print(f"\nQuestion: {question}")
        print("\nTop 5 results:")
        
        results = embeddings.semantic_search(question, top_k=5)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['name']} ({r['type']}) - similarity: {r['similarity']:.3f}")
            print(f"   {r['description']}")
    else:
        print("\nCancelled.")

if __name__ == "__main__":
    main()
