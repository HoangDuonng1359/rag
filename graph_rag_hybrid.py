"""
Graph RAG 
Kết hợp Vector Search (Bước 2) và Graph Traversal (Bước 1)
để tạo retrieval system mạnh mẽ và chính xác
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np


class HybridRetriever:
    """
    Hybrid Retrieval: Combine semantic search với graph traversal
    """
    
    def __init__(self, graph_query, embeddings):
        """
        Initialize với graph query và embeddings instances
        
        Args:
            graph_query: GraphRAGQuery instance (Bước 1)
            embeddings: EntityEmbeddings instance (Bước 2)
        """
        self.graph_query = graph_query
        self.embeddings = embeddings
        
        # Question type keywords để infer intent
        self.question_patterns = {
            'WHO': ['ai', 'người nào', 'cá nhân', 'tổ chức', 'chủ xe'],
            'WHERE': ['ở đâu', 'nơi nào', 'địa điểm', 'khu vực', 'vùng'],
            'WHEN': ['khi nào', 'năm nào', 'thời gian', 'lúc nào'],
            'WHAT': ['gì', 'vi phạm', 'quy định', 'điều luật', 'biển báo'],
            'WHY': ['tại sao', 'vì sao', 'lý do', 'nguyên nhân'],
            'HOW': ['như thế nào', 'bằng cách nào', 'thế nào']
        }
        
        # Entity type weights cho từng question type
        self.type_weights = {
            'WHO': {'PERSON': 1.0, 'ORGANIZATION': 0.7, 'EVENT': 0.3},
            'WHERE': {'LOCATION': 1.0, 'EVENT': 0.5, 'ORGANIZATION': 0.3},
            'WHEN': {'TIME': 1.0, 'EVENT': 0.8, 'PERSON': 0.2},
            'WHAT': {'EVENT': 1.0, 'ORGANIZATION': 0.6, 'PERSON': 0.4},
            'DEFAULT': {'PERSON': 0.5, 'LOCATION': 0.5, 'ORGANIZATION': 0.5, 
                       'EVENT': 0.8, 'TIME': 0.5}
        }
    
    def infer_question_type(self, question: str) -> str:
        """
        Infer question type từ keywords
        
        Args:
            question: User question
            
        Returns:
            Question type (WHO, WHERE, WHEN, WHAT, WHY, HOW, DEFAULT)
        """
        question_lower = question.lower()
        
        for q_type, keywords in self.question_patterns.items():
            for keyword in keywords:
                if keyword in question_lower:
                    return q_type
        
        return 'DEFAULT'
    
    def retrieve(
        self,
        question: str,
        top_k: int = 15,
        vector_top_k: int = 5,
        expansion_depth: int = 1,
        similarity_threshold: float = 0.6,
        include_paths: bool = False
    ) -> Dict:
        """
        Main hybrid retrieval function
        
        Args:
            question: User question
            top_k: Số lượng entities cuối cùng
            vector_top_k: Số entities từ vector search
            expansion_depth: Độ sâu graph expansion (1 or 2)
            similarity_threshold: Min similarity score để expand
            include_paths: Include paths between entities
            
        Returns:
            Dict với retrieved context
        """
        # Infer question type
        q_type = self.infer_question_type(question)
        
        # Vector search for seed entities
        seed_entities = self.embeddings.semantic_search(
            question, 
            top_k=vector_top_k
        )
        
        # print(f"   Found {len(seed_entities)} seed entities:")
        # for i, entity in enumerate(seed_entities[:3], 1):
        #     print(f"   {i}. {entity['name']} ({entity['type']}) - {entity['similarity']:.3f}")
        # if len(seed_entities) > 3:
        #     print(f"   ... and {len(seed_entities) - 3} more")
        
        # Filter by similarity threshold
        high_confidence_seeds = [
            e for e in seed_entities 
            if e['similarity'] >= similarity_threshold
        ]
        
        if not high_confidence_seeds:
            # print(f" No entities above threshold {similarity_threshold}")
            high_confidence_seeds = seed_entities[:2]  # At least take top 2
        
        # Graph expansion
        # print(f"\n GRAPH EXPANSION (depth={expansion_depth})...")
        expanded_entities = self._expand_graph(
            high_confidence_seeds,
            expansion_depth
        )
        
        # print(f"   Expanded to {len(expanded_entities)} total entities")
        
        # Hybrid scoring và ranking
        # print(f"\n HYBRID SCORING & RANKING...")
        scored_entities = self._hybrid_score(
            seed_entities=seed_entities,
            expanded_entities=expanded_entities,
            question=question,
            q_type=q_type
        )
        
        # Deduplicate và sort
        unique_entities = self._deduplicate_entities(scored_entities)
        ranked = sorted(unique_entities, key=lambda x: x['score'], reverse=True)
        top_entities = ranked[:top_k]
        
        print(f"   Top {min(top_k, len(top_entities))} entities selected")
        
        # Extract relationships
        relationships = self._extract_relationships(top_entities)
        # print(f"   Found {len(relationships)} relationships")
        
        # Find paths (optional)
        paths = []
        if include_paths and len(top_entities) >= 2:
         #   print(f"\n FINDING PATHS...")
            paths = self._find_key_paths(top_entities[:5])
         #   print(f"   Found {len(paths)} paths")
        
        # Build final context
        context = {
            'question': question,
            'question_type': q_type,
            'top_entities': top_entities,
            'relationships': relationships,
            'paths': paths,
            'retrieval_stats': {
                'seed_entities': len(seed_entities),
                'expanded_entities': len(expanded_entities),
                'final_entities': len(top_entities),
                'relationships': len(relationships)
            }
        }
        
        return context
    
    def _expand_graph(
        self,
        seed_entities: List[Dict],
        max_depth: int
    ) -> List[Dict]:
        """
        Expand graph từ seed entities
        
        Args:
            seed_entities: Starting entities từ vector search
            max_depth: Maximum expansion depth
            
        Returns:
            List of expanded entities
        """
        expanded = []
        seen_names = set()
        
        for i, seed in enumerate(seed_entities):
            entity_name = seed['name']
            
            # Determine depth: top entity gets more expansion
            depth = max_depth if i == 0 else 1
            
            # Get related entities
            related = self.graph_query.get_related_entities(
                entity_name,
                max_depth=depth,
                limit=30
            )
            
            for rel_entity in related:
                name = rel_entity['name']
                if name not in seen_names:
                    seen_names.add(name)
                    expanded.append({
                        'name': name,
                        'type': rel_entity['type'],
                        'description': rel_entity['description'],
                        'distance': rel_entity['distance'],
                        'source_seed': entity_name,
                        'source_similarity': seed['similarity']
                    })
        
        return expanded
    
    def _hybrid_score(
        self,
        seed_entities: List[Dict],
        expanded_entities: List[Dict],
        question: str,
        q_type: str
    ) -> List[Dict]:
        """
        Calculate hybrid scores combining multiple signals
        
        Args:
            seed_entities: Entities từ vector search
            expanded_entities: Entities từ graph expansion
            question: Original question
            q_type: Question type
            
        Returns:
            List of entities với hybrid scores
        """
        # Create lookup for seed entities
        seed_lookup = {e['name']: e['similarity'] for e in seed_entities}
        
        # Get type weights cho question type
        type_weights = self.type_weights.get(q_type, self.type_weights['DEFAULT'])
        
        scored = []
        
        # Score expanded entities
        for entity in expanded_entities:
            # Vector score (if in seeds)
            vector_score = seed_lookup.get(entity['name'], 0.0)
            
            # Graph score (based on distance)
            distance = entity.get('distance', 3)
            graph_score = 1.0 / (distance + 1)
            
            # 3. Type score (based on question type)
            entity_type = entity['type']
            type_score = type_weights.get(entity_type, 0.3)
            
            # 4. Source score (from seed similarity)
            source_score = entity.get('source_similarity', 0.5)
            
            # Combined hybrid score
            hybrid_score = (
                0.35 * vector_score +      # Direct semantic match
                0.25 * graph_score +        # Graph proximity
                0.25 * type_score +         # Type relevance
                0.15 * source_score        # Seed quality
            )
            
            scored.append({
                'name': entity['name'],
                'type': entity['type'],
                'description': entity['description'],
                'score': hybrid_score,
                'vector_score': vector_score,
                'graph_score': graph_score,
                'type_score': type_score,
                'distance': distance
            })
        
        # add seed entities với full scores
        for seed in seed_entities:
            if seed['name'] not in [e['name'] for e in scored]:
                type_score = type_weights.get(seed['type'], 0.5)
                scored.append({
                    'name': seed['name'],
                    'type': seed['type'],
                    'description': seed['description'],
                    'score': 0.5 * seed['similarity'] + 0.5 * type_score,
                    'vector_score': seed['similarity'],
                    'graph_score': 1.0,
                    'type_score': type_score,
                    'distance': 0,
                    'first_seen_page': seed.get('first_seen_page'),
                    'first_seen_chapter': seed.get('first_seen_chapter')
                })
        
        return scored
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Remove duplicate entities, keeping highest score
        
        Args:
            entities: List of entities với scores
            
        Returns:
            Deduplicated list
        """
        best_scores = {}
        
        for entity in entities:
            name = entity['name']
            score = entity['score']
            
            if name not in best_scores or score > best_scores[name]['score']:
                best_scores[name] = entity
        
        return list(best_scores.values())
    
    def _extract_relationships(self, entities: List[Dict]) -> List[Dict]:
        """
        Extract relationships between selected entities
        
        Args:
            entities: List of top entities
            
        Returns:
            List of relationships
        """
        entity_names = {e['name'] for e in entities}
        relationships = []
        seen_rels = set()
        important_types = {"PENALTY", "VIOLATION", "REGULATION"}
        
        for entity in entities[:10]:  # Limit to top 10 to avoid too many queries
            # Get neighborhood
            neighborhood = self.graph_query.get_entity_neighborhood(
                entity['name']
            )
            
            if not neighborhood or 'relationships' not in neighborhood:
                continue
            
            # Extract relationships to other selected entities
            for rel_type, connections in neighborhood['relationships'].items():
                for conn in connections:
                    target = conn['entity']
                    target_type = conn.get('entity_type')
                    
                    # Include if target is selected OR target is an important type (e.g., PENALTY/VIOLATION/REGULATION)
                    if target in entity_names or (target_type in important_types):
                        rel_key = (entity['name'], target, rel_type)
                        
                        if rel_key not in seen_rels:
                            seen_rels.add(rel_key)
                            relationships.append({
                                'source': entity['name'],
                                'target': target,
                                'type': rel_type,
                                'description': conn.get('description', ''),
                                'direction': conn.get('direction', 'outgoing'),
                                # Pass through law metadata when available
                                'law_name': conn.get('law_name'),
                                'law_code': conn.get('law_code'),
                                'chapter': conn.get('chapter'),
                                'chapter_title': conn.get('chapter_title'),
                                'article': conn.get('article'),
                                'clauses': conn.get('clauses'),
                                'has_penalty': conn.get('has_penalty'),
                                'mode': conn.get('mode')
                            })
        
        return relationships
    
    def _find_key_paths(self, entities: List[Dict]) -> List[Dict]:
        """
        Find important paths between top entities
        
        Args:
            entities: Top entities
            
        Returns:
            List of paths
        """
        paths = []
        
        # Only find paths between top 5 entities
        top_5 = entities[:5]
        
        for i, source in enumerate(top_5):
            for target in top_5[i+1:]:
                path_results = self.graph_query.find_paths_between_entities(
                    source['name'],
                    target['name'],
                    max_depth=3
                )
                
                if path_results:
                    # Take shortest path
                    shortest = path_results[0]
                    paths.append({
                        'source': source['name'],
                        'target': target['name'],
                        'path_length': shortest['path_length'],
                        'nodes': shortest['nodes'],
                        'relationships': shortest['relationships']
                    })
        
        return paths
    
    def retrieve_simple(
        self,
        question: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Simplified retrieval - chỉ return top entities
        
        Args:
            question: User question
            top_k: Number of results
            
        Returns:
            List of top entities
        """
        context = self.retrieve(
            question=question,
            top_k=top_k,
            vector_top_k=5,
            expansion_depth=1,
            include_paths=False
        )
        
        return context['top_entities']
    
    def format_context_for_llm(self, context: Dict, max_entities: int = 10) -> str:
        """
        Format retrieved context thành text cho LLM prompt
        
        Args:
            context: Context dict từ retrieve()
            max_entities: Max entities to include
            
        Returns:
            Formatted context string
        """
        lines = []
        
        lines.append("=== RETRIEVED CONTEXT ===\n")
        
        # Entities
        lines.append("ENTITIES:")
        for i, entity in enumerate(context['top_entities'][:max_entities], 1):
            lines.append(f"{i}. {entity['name']} ({entity['type']})")
            if entity.get('description'):
                lines.append(f"   {entity['description']}")
            lines.append(f"   [Relevance: {entity['score']:.3f}]")
            lines.append("")
        
        # Relationships
        if context['relationships']:
            lines.append("\nRELATIONSHIPS:")
            for i, rel in enumerate(context['relationships'][:15], 1):
                arrow = "→" if rel['direction'] == 'outgoing' else "←"
                lines.append(
                    f"{i}. {rel['source']} {arrow} [{rel['type']}] {arrow} {rel['target']}"
                )
                if rel.get('description'):
                    lines.append(f"   {rel['description']}")
            lines.append("")
        
        # Paths (if available)
        if context.get('paths'):
            lines.append("\nKEY CONNECTIONS:")
            for i, path in enumerate(context['paths'][:5], 1):
                path_str = " → ".join([n['name'] for n in path['nodes']])
                lines.append(f"{i}. {path_str} (length: {path['path_length']})")
            lines.append("")
        
        return "\n".join(lines)
    
    def get_statistics(self, context: Dict) -> Dict:
        """
        Get statistics về retrieval results
        
        Args:
            context: Context dict từ retrieve()
            
        Returns:
            Statistics dict
        """
        entities = context['top_entities']
        
        # Count by type
        type_counts = defaultdict(int)
        for entity in entities:
            type_counts[entity['type']] += 1
        
        # Average scores
        scores = [e['score'] for e in entities]
        avg_score = np.mean(scores) if scores else 0
        
        return {
            'total_entities': len(entities),
            'entity_types': dict(type_counts),
            'avg_score': float(avg_score),
            'relationships': len(context['relationships']),
            'paths': len(context.get('paths', [])),
            'question_type': context['question_type']
        }


# Example usage
if __name__ == "__main__":
    from langchain_community.graphs import Neo4jGraph
    from graph_rag_query import GraphRAGQuery
    from graph_rag_embeddings import EntityEmbeddings
    
    # Setup
    NEO4J_URI = "neo4j+s://0c367113.databases.neo4j.io"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "gTO1K567hBLzkRdUAhhEb-UqvBjz0i3ckV3M9v_-Nio"
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize components
    graph_query = GraphRAGQuery(graph)
    embeddings = EntityEmbeddings(graph)
    
    # Initialize hybrid retriever
    hybrid = HybridRetriever(graph_query, embeddings)
    
    # Test retrieval
    question = "Quy định về tốc độ tối đa trong khu dân cư là gì?"
    context = hybrid.retrieve(question, top_k=10)
    
    # Print formatted context
    print("\n" + "="*70)
    formatted = hybrid.format_context_for_llm(context)
    print(formatted)
    
    # Print statistics
    stats = hybrid.get_statistics(context)
    print("\n=== STATISTICS ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
