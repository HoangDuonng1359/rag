"""
Graph Query Functions cho RAG System
Các hàm query Neo4j để retrieve entities và relationships
"""

from typing import List, Dict, Optional, Tuple
from langchain_community.graphs import Neo4jGraph


class GraphRAGQuery:
    """Class chứa các query functions cho Graph RAG"""
    
    def __init__(self, graph: Neo4jGraph):
        """
        Initialize với Neo4j graph connection
        
        Args:
            graph: Neo4jGraph instance đã kết nối
        """
        self.graph = graph
        self._create_indexes()
    
    def _create_indexes(self):
        """Tạo indexes cho Neo4j để tối ưu query performance"""
        index_queries = [
            # Index cho name property trên tất cả node types
            "CREATE INDEX entity_name IF NOT EXISTS FOR (n:PERSON) ON (n.name)",
            "CREATE INDEX location_name IF NOT EXISTS FOR (n:LOCATION) ON (n.name)",
            "CREATE INDEX org_name IF NOT EXISTS FOR (n:ORGANIZATION) ON (n.name)",
            "CREATE INDEX event_name IF NOT EXISTS FOR (n:EVENT) ON (n.name)",
            "CREATE INDEX time_name IF NOT EXISTS FOR (n:TIME) ON (n.name)",
        ]
        
        for query in index_queries:
            try:
                self.graph.query(query)
            except Exception as e:
                print(f"Index already exists or error: {e}")
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Dict]:
        """
        Tìm entity theo tên chính xác
        
        Args:
            name: Tên entity cần tìm
            entity_type: Loại entity (PERSON, LOCATION, etc.) - optional
            
        Returns:
            Dict chứa thông tin entity hoặc None
        """
        if entity_type:
            query = f"""
            MATCH (n:{entity_type} {{name: $name}})
            RETURN n.name as name, 
                   labels(n)[0] as type,
                   n.description as description,
                   n.first_seen_chapter as chapter,
                   n.first_seen_page as page
            LIMIT 1
            """
        else:
            query = """
            MATCH (n {name: $name})
            RETURN n.name as name,
                   labels(n)[0] as type,
                   n.description as description,
                   n.first_seen_chapter as chapter,
                   n.first_seen_page as page
            LIMIT 1
            """
        
        result = self.graph.query(query, {"name": name})
        return result[0] if result else None
    
    def search_entities_by_name(self, search_term: str, limit: int = 10) -> List[Dict]:
        """
        Tìm kiếm entities có tên chứa search term (case-insensitive)
        
        Args:
            search_term: Từ khóa tìm kiếm
            limit: Số lượng kết quả tối đa
            
        Returns:
            List các entities khớp
        """
        query = """
        MATCH (n)
        WHERE toLower(n.name) CONTAINS toLower($search_term)
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description,
               n.first_seen_page as first_seen_page,
               n.first_seen_chapter as first_seen_chapter
        LIMIT $limit
        """
        
        return self.graph.query(query, {"search_term": search_term, "limit": limit})
    
    def get_related_entities(
        self, 
        entity_name: str, 
        max_depth: int = 1,
        limit: int = 20
    ) -> List[Dict]:
        """
        Lấy các entities liên quan trực tiếp hoặc gián tiếp
        
        Args:
            entity_name: Tên entity gốc
            max_depth: Độ sâu tối đa của relationship (1 = trực tiếp, 2 = gián tiếp)
            limit: Số lượng kết quả tối đa
            
        Returns:
            List các entities liên quan với relationship info
        """
        query = f"""
        MATCH path = (source {{name: $entity_name}})-[r*1..{max_depth}]-(related)
        WITH related, 
             relationships(path) as rels,
             length(path) as distance
        RETURN DISTINCT related.name as name,
               labels(related)[0] as type,
               related.description as description,
               related.first_seen_page as first_seen_page,
               related.first_seen_chapter as first_seen_chapter,
               [rel in rels | {{
                   type: type(rel),
                   description: rel.description
               }}] as relationships,
               distance
        ORDER BY distance, related.name
        LIMIT $limit
        """
        
        return self.graph.query(query, {"entity_name": entity_name, "limit": limit})
    
    def find_paths_between_entities(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 4
    ) -> List[Dict]:
        """
        Tìm tất cả đường đi giữa 2 entities
        
        Args:
            source_name: Entity nguồn
            target_name: Entity đích
            max_depth: Độ dài tối đa của path
            
        Returns:
            List các paths với nodes và relationships
        """
        query = f"""
        MATCH path = shortestPath(
            (source {{name: $source_name}})-[*1..{max_depth}]-(target {{name: $target_name}})
        )
        WITH path, length(path) as path_length
        RETURN [node in nodes(path) | {{
                   name: node.name,
                   type: labels(node)[0],
                   description: node.description
               }}] as nodes,
               [rel in relationships(path) | {{
                   type: type(rel),
                   description: rel.description
               }}] as relationships,
               path_length
        ORDER BY path_length
        LIMIT 5
        """
        
        return self.graph.query(query, {
            "source_name": source_name,
            "target_name": target_name
        })
    
    def get_entity_neighborhood(
        self,
        entity_name: str,
        include_relationships: bool = True
    ) -> Dict:
        """
        Lấy toàn bộ neighborhood của một entity (1-hop)
        
        Args:
            entity_name: Tên entity
            include_relationships: Include relationship details
            
        Returns:
            Dict chứa entity info, connected entities, và relationships
        """
        # Get entity info
        entity_query = """
        MATCH (n {name: $entity_name})
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description,
               n.first_seen_chapter as chapter,
               n.first_seen_page as page
        """
        entity_info = self.graph.query(entity_query, {"entity_name": entity_name})
        
        if not entity_info:
            return {"entity": None, "connected_entities": [], "relationships": []}
        
        # Get connected entities và relationships
        neighborhood_query = """
        MATCH (source {name: $entity_name})-[r]-(connected)
        RETURN connected.name as name,
               labels(connected)[0] as type,
               connected.description as description,
               type(r) as relationship_type,
               r.description as relationship_description,
               CASE WHEN startNode(r).name = $entity_name THEN 'outgoing' ELSE 'incoming' END as direction
        """
        
        connected = self.graph.query(neighborhood_query, {"entity_name": entity_name})
        
        # Group by relationship type
        relationships_by_type = {}
        connected_entities = []
        
        for conn in connected:
            rel_type = conn['relationship_type']
            if rel_type not in relationships_by_type:
                relationships_by_type[rel_type] = []
            
            relationships_by_type[rel_type].append({
                "entity": conn['name'],
                "entity_type": conn['type'],
                "direction": conn['direction'],
                "description": conn['relationship_description']
            })
            
            connected_entities.append({
                "name": conn['name'],
                "type": conn['type'],
                "description": conn['description']
            })
        
        return {
            "entity": entity_info[0],
            "connected_entities": connected_entities,
            "relationships": relationships_by_type
        }
    
    def get_entities_by_type(self, entity_type: str, limit: int = 50) -> List[Dict]:
        """
        Lấy tất cả entities theo loại
        
        Args:
            entity_type: PERSON, LOCATION, ORGANIZATION, EVENT, TIME
            limit: Số lượng tối đa
            
        Returns:
            List entities
        """
        query = f"""
        MATCH (n:{entity_type})
        RETURN n.name as name,
               n.description as description,
               n.first_seen_chapter as chapter,
               n.first_seen_page as page
        ORDER BY n.name
        LIMIT $limit
        """
        
        return self.graph.query(query, {"limit": limit})
    
    def get_most_connected_entities(self, limit: int = 20) -> List[Dict]:
        """
        Lấy các entities có nhiều connections nhất (hubs)
        
        Args:
            limit: Số lượng kết quả
            
        Returns:
            List entities với connection count
        """
        query = """
        MATCH (n)-[r]-()
        WITH n, count(DISTINCT r) as connection_count
        WHERE connection_count > 0
        RETURN n.name as name,
               labels(n)[0] as type,
               n.description as description,
               connection_count
        ORDER BY connection_count DESC
        LIMIT $limit
        """
        
        return self.graph.query(query, {"limit": limit})
    
    def get_relationship_statistics(self) -> Dict:
        """
        Lấy thống kê về relationships trong graph
        
        Returns:
            Dict với statistics
        """
        # Count by type
        type_query = """
        MATCH ()-[r]->()
        RETURN type(r) as relationship_type, count(r) as count
        ORDER BY count DESC
        """
        
        # Total counts
        total_query = """
        MATCH (n)
        WITH count(n) as node_count
        MATCH ()-[r]->()
        RETURN node_count, count(r) as relationship_count
        """
        
        types = self.graph.query(type_query)
        totals = self.graph.query(total_query)[0]
        
        return {
            "total_nodes": totals['node_count'],
            "total_relationships": totals['relationship_count'],
            "relationship_types": types
        }


# Example usage
if __name__ == "__main__":
    from langchain_community.graphs import Neo4jGraph
    
    # Setup connection
    NEO4J_URI = "neo4j+s://0c367113.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "gTO1K567hBLzkRdUAhhEb-UqvBjz0i3ckV3M9v_-Nio"
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize query class
    rag_query = GraphRAGQuery(graph)
    
    # Test queries
    print("=== Test 1: Search entities ===")
    results = rag_query.search_entities_by_name("Việt", limit=5)
    for r in results:
        print(f"- {r['name']} ({r['type']})")
    
    print("\n=== Test 2: Get statistics ===")
    stats = rag_query.get_relationship_statistics()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Total relationships: {stats['total_relationships']}")
    
    print("\n=== Test 3: Most connected entities ===")
    hubs = rag_query.get_most_connected_entities(limit=10)
    for hub in hubs:
        print(f"- {hub['name']}: {hub['connection_count']} connections")
