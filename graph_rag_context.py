from typing import List, Dict, Optional
from datetime import datetime
from context_rag import ContextRAG
import re
import json
import os


class ContextBuilder:
    """
    Build và format context từ hybrid retrieval results thành prompts cho LLM
    Optimized cho Gemini API
    """
    
    def __init__(self, max_context_length: int = 8000, chunks_file: Optional[str] = None, include_law_content: bool = True):
        """
        Initialize context builder
        
        Args:
            max_context_length: Max characters cho context (để avoid token limits)
            chunks_file: Path to chunks_by_clause.jsonl file
            include_law_content: Có thêm nội dung luật chi tiết vào relationships hay không
        """
        self.max_context_length = max_context_length
        self.include_law_content = include_law_content
        #self.rag_context = ContextRAG()
        
        # Load chunks from file
        if chunks_file is None:
            # Default path
            chunks_file = os.path.join(
                os.path.dirname(__file__), 
                'data', 'chunk', 'chunks_by_clause.jsonl'
            )
        
        self.chunks_dict = self._load_chunks(chunks_file)
        print(f"Loaded {len(self.chunks_dict)} chunks from {chunks_file}")
    
    def _load_chunks(self, chunks_file: str) -> Dict[str, Dict]:
        """Load chunks từ JSONL file và index bằng ID"""
        chunks = {}
        
        if not os.path.exists(chunks_file):
            print(f"Warning: Chunks file not found: {chunks_file}")
            return chunks
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line.strip())
                    chunks[chunk['id']] = chunk
        except Exception as e:
            print(f"Error loading chunks: {e}")
        
        return chunks
    
    def _find_chunk_content(self, law_code: str, article: int, clauses: Optional[List[int]] = None) -> Optional[str]:
        """
        Tìm chunk content dựa trên law_code, article, và clauses
        
        Args:
            law_code: Law code
            article: Article number
            clauses: List of clause numbers
            
        Returns:
            Chunk content hoặc None
        """
        if not law_code or article is None:
            return None
        
        # Convert law_code: replace / with _
        law_id = law_code.replace('/', '_')
        
        # Build base prefix
        base_prefix = f"{law_id}_art{article}"
        
        # Try to find matching chunks
        matching_chunks = []
        
        if clauses and len(clauses) > 0:
            # Try to find chunks with specific clauses
            for clause in clauses:
                # Pattern: law_code_art{article}_*_khoan{clause} or law_code_art{article}_khoan{clause}
                # This will match: 168_2024_NĐ-CP_art5_khoan2, 168_2024_NĐ-CP_art5_p2_khoan2, etc.
                for chunk_id, chunk in self.chunks_dict.items():
                    if chunk_id.startswith(base_prefix) and f"_khoan{clause}" in chunk_id:
                        matching_chunks.append(chunk)
                        break  # Found one chunk for this clause
        
        # If no clause-specific chunks found, try article-level chunks
        if not matching_chunks:
            for chunk_id, chunk in self.chunks_dict.items():
                # Exact match or starts with base_prefix and next char is not digit (to avoid art5 matching art50)
                if chunk_id == base_prefix or (chunk_id.startswith(base_prefix) and 
                                              len(chunk_id) > len(base_prefix) and 
                                              chunk_id[len(base_prefix)] in ['_', 'p']):
                    matching_chunks.append(chunk)
        
        # Combine content from matching chunks
        if matching_chunks:
            # Sort by chunk_index if available
            matching_chunks.sort(key=lambda x: x.get('metadata', {}).get('chunk_index', 0))
            # Combine content
            combined_content = " ".join(chunk['content'] for chunk in matching_chunks)
            return combined_content
        
        return None
    
    def build_rag_context(
        self,
        question: str,
        retrieval_context: Dict,
        max_entities: int = 12,
        max_relationships: int = 20,
        include_paths: bool = True,
        include_sources: bool = True,
    ) -> Dict:
        """
        Build complete RAG context từ retrieval results
        
        Args:
            question: User question
            retrieval_context: Output từ HybridRetriever.retrieve()
            max_entities: Max entities to include
            max_relationships: Max relationships to include
            include_paths: Include paths between entities
            include_sources: Include source citations
            
        Returns:
            Dict với structured context
        """
        entities = retrieval_context['top_entities'][:max_entities]
        relationships = retrieval_context['relationships'][:max_relationships]
        paths = retrieval_context.get('paths', [])[:5] if include_paths else []
        chunks = retrieval_context.get('chunks', [])  # Lấy chunks từ retrieval context
        
        # Extract source citations với content từ chunks
        sources = self._extract_sources_from_chunks(chunks, entities) if include_sources else []
        
        # Build context sections
        context = {
            'question': question,
            'question_type': retrieval_context.get('question_type', 'DEFAULT'),
            'entities': self._format_entities(entities),
            'relationships': self._format_relationships(relationships),
            'paths': self._format_paths(paths) if paths else None,
            'sources': sources,  # Bây giờ là list of dicts với content
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'context_summary': self._create_summary(entities, relationships)
        }
        
        return context
    
    def _format_entities(self, entities: List[Dict]) -> List[Dict]:
        """Format entities thành structured format"""
        formatted = []
        
        for entity in entities:
            formatted.append({
                'name': entity['name'],
                'type': entity['type'],
                'description': entity.get('description', ''),
                'relevance_score': round(entity['score'], 3)
            })
        
        return formatted
    
    def _format_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Format relationships thành structured format"""
        formatted = []
        
        for rel in relationships:
            formatted.append({
                'source': rel['source'],
                'relation': rel['type'],
                'target': rel['target'],
                'description': rel.get('description', ''),
                'direction': rel.get('direction', 'outgoing'),
                # Relationship metadata from graph
                'mode': rel.get('mode'),
                'law_name': rel.get('law_name'),
                'law_code': rel.get('law_code'),
                'chapter': rel.get('chapter'),
                'chapter_title': rel.get('chapter_title'),
                'article': rel.get('article'),
                'clauses': rel.get('clauses', []),
                'has_penalty': rel.get('has_penalty')
            })
        
        return formatted
    
    def _format_paths(self, paths: List[Dict]) -> List[Dict]:
        """Format paths between entities"""
        formatted = []
        
        for path in paths:
            path_nodes = [node['name'] for node in path['nodes']]
            path_rels = [rel['type'] for rel in path['relationships']]
            
            formatted.append({
                'from': path['source'],
                'to': path['target'],
                'length': path['path_length'],
                'route': path_nodes,
                'connections': path_rels
            })
        
        return formatted
    
    def _extract_sources_from_chunks(self, chunks: List[Dict], entities: List[Dict]) -> List[Dict]:
        """Extract source citations từ chunks và entities"""
        sources = []
        
        # Từ chunks (nếu có)
        seen_sources = set()
        for chunk in chunks[:5]:  # Limit 5 chunks quan trọng nhất
            content = chunk.get('content', '')
            metadata = chunk.get('metadata', {})
            
            law_code = metadata.get('law_code', 'N/A')
            article = metadata.get('article', 'N/A')
            chapter = metadata.get('chapter', 'N/A')
            law_name = metadata.get('law_name', '')
            
            source_key = f"{law_code}_art{article}"
            if source_key not in seen_sources:
                sources.append({
                    'law_code': law_code,
                    'law_name': law_name,
                    'article': article,
                    'chapter': chapter,
                    'citation': f"{law_code}, Điều {article}",
                    'content': content[:1000]  # Limit content length
                })
                seen_sources.add(source_key)
        
        # Nếu không có chunks, lấy từ entities
        if not sources:
            for entity in entities[:3]:
                law_code = entity.get('law_code', 'N/A')
                article = entity.get('article', 'N/A')
                
                if law_code != 'N/A':
                    source_key = f"{law_code}_art{article}"
                    if source_key not in seen_sources:
                        sources.append({
                            'law_code': law_code,
                            'law_name': entity.get('law_name', ''),
                            'article': article,
                            'chapter': entity.get('chapter', 'N/A'),
                            'citation': f"{law_code}, Điều {article}",
                            'content': entity.get('description', '')
                        })
                        seen_sources.add(source_key)
        
        return sources
    
    def _create_summary(self, entities: List[Dict], relationships: List[Dict]) -> str:
        """Tạo brief summary về context"""
        entity_types = {}
        for e in entities:
            etype = e['type']
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        summary_parts = []
        for etype, count in entity_types.items():
            summary_parts.append(f"{count} {etype}")
        
        summary = f"Context includes {', '.join(summary_parts)} với {len(relationships)} relationships"
        return summary
    
    def format_for_gemini(
        self,
        context: Dict,
        prompt_type: str = "qa",
        include_instructions: bool = True,
        include_law_content: Optional[bool] = None,
        tranditional_rag_context : ContextRAG = None
    ) -> str:
        """
        Args:
            context: Context từ build_rag_context()
            prompt_type: Loại prompt (qa, summary, explain, timeline)
            include_instructions: Include system instructions
            include_law_content: Có thêm nội dung luật chi tiết vào relationships (None = dùng default từ __init__)
            
        Returns:
            Formatted prompt string
        """
        # Use parameter if provided, otherwise use instance default
        if include_law_content is None:
            include_law_content = self.include_law_content
        
        return self._format_qa_prompt(context, include_instructions, include_law_content, rag_context=tranditional_rag_context)
    
    def _format_qa_prompt(self, context: Dict, include_instructions: bool, include_law_content: bool = True, 
                          rag_context : ContextRAG = None ) -> str:
        """Format prompt cho Q&A task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông, Hãy đóng vai là một nhân viên tư vấn luật giao thông. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên dữ liệu luật được cung cấp.

HƯỚNG DẪN:
- Chỉ sử dụng thông tin từ context được cung cấp
- Trả lời chính xác, có căn cứ
- Nếu không có đủ thông tin, hãy nói rõ
- Sử dụng tiếng Việt tự nhiên và dễ hiểu, không cần chào hỏi
YÊU CẦU TRẢ LỜI:
- Trả lời theo dạng đoạn văn mạch lạc
- Không trả lời nếu thông tin quá mơ hồ hoặc không liên quan
- Tránh sử dụng các cụm từ như "theo ngữ cảnh đã cho", "dựa trên thông tin được cung cấp"
- Nêu rõ:
  + Quy định / nội dung chính
  + Dẫn chứng (nếu có)
  + Giải thích
- Phải có kết luận cuối cùng (kết luận highlight lên để người đọc dễ nhận biết)
- Đoạn cuối ghi các nguồn(nếu có, nếu không có, bạn không sử dụng dữ liệu từ các nguồn thì không thêm vào): Ví dụ: Nguồn: Nghị định 168/2024/NĐ-CP, Điều 5, Khoản 2, Luật Trật tự, an toàn giao thông đường bộ \n Nghị định 100/2019/NĐ-CP, Điều 6, Khoản 1.

""")
        parts.append("CONTEXT:")
        parts.append("")
        
        # Entities section
        parts.append("ENTITIES:")
        for i, entity in enumerate(context['entities'], 1):
            parts.append(f"\n{i}. {entity['name']} ({entity['type']})")
            if entity['description']:
                parts.append(f"   Mô tả: {entity['description']}")
        
        # Relationships section
        parts.append("\nRELATIONSHIPS:")
        for i, rel in enumerate(context['relationships'], 1):
            arrow = "->" if rel['direction'] == 'outgoing' else "<-"
            parts.append(f"\n{i}. {rel['source']} {arrow} [{rel['relation']}] {arrow} {rel['target']}")
            if rel['description']:
                parts.append(f"   Chi tiết: {rel['description']}")
            
            # Show relationship metadata if available
            law_code = rel.get('law_code')
            law_name = rel.get('law_name', '')
            article = rel.get('article')
            clauses = rel.get('clauses')
            
            if law_code or law_name:
                parts.append(f"   Nguồn luật: {law_code or 'N/A'}{' - ' + law_name if law_name else ''}")
            
            if article is not None:
                parts.append(f"   Điều: {article}")
            
            if rel.get('chapter') or rel.get('chapter_title'):
                chapter = rel.get('chapter', '')
                chapter_title = rel.get('chapter_title', '')
                parts.append(f"   Chương: {chapter}{' - ' + chapter_title if chapter_title else ''}")
            
            if rel.get('mode'):
                parts.append(f"   Mode: {rel.get('mode')}")
            
            if clauses:
                if isinstance(clauses, list):
                    clause_str = ", ".join(str(c) for c in clauses)
                else:
                    clause_str = str(clauses)
                parts.append(f"   Khoản: {clause_str}")
            # cộng thêm vector search 
        rag_result = rag_context.rag_retrieve(
            query=context['question'],
            top_k_dense=10,
            use_rerank=False,
            top_n_final=10,
        )
        context_from_vector_search = rag_context.build_llm_context_from_hits(rag_result['hits'])
        
        parts.append("-"*10)
        parts.append("\nVĂN BẢN LUẬT LIÊN QUAN:")
        parts.append("\n" + context_from_vector_search)

        parts.append("\n" + "-" * 10)
        if context.get('paths'):
            parts.append("\nCONNECTIONS (Kết nối):")
            for i, path in enumerate(context['paths'], 1):
                route_str = " → ".join(path['route'])
                parts.append(f"\n{i}. {path['from']} đến {path['to']}: {route_str}")
        
        parts.append("\n" + "-" * 10)
        parts.append(f"\nCÂU HỎI: {context['question']}")
        
        full_prompt = "\n".join(parts)
        
        return full_prompt
    
    
    def truncate_to_token_limit(
        self,
        context: Dict,
        max_tokens: int = 6000
    ) -> Dict:
        """
        Truncate context để fit token limit
        
        Args:
            context: Full context
            max_tokens: Max tokens allowed
            
        Returns:
            Truncated context
        """
        # Start with most important elements
        truncated = context.copy()
        
        # Gradually reduce less important elements
        while True:
            # Format và estimate tokens
            formatted = self.format_for_gemini(truncated, prompt_type="qa", tranditional_rag_context=None)
            # Reduce elements
            if len(truncated['relationships']) > 10:
                truncated['relationships'] = truncated['relationships'][:10]
            elif len(truncated['entities']) > 8:
                truncated['entities'] = truncated['entities'][:8]
            elif truncated.get('paths'):
                truncated['paths'] = None
            else:
                # Last resort: truncate entity descriptions
                for entity in truncated['entities']:
                    if entity['description']:
                        entity['description'] = entity['description'][:100]
                break
        
        return truncated