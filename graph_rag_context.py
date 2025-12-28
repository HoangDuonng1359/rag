"""
Graph RAG - Context Builder
Format retrieved context thành structured prompts cho LLM (Gemini)
"""

from typing import List, Dict, Optional
from datetime import datetime
import re


class ContextBuilder:
    """
    Build và format context từ hybrid retrieval results thành prompts cho LLM
    Optimized cho Gemini API
    """
    
    def __init__(self, max_context_length: int = 8000):
        """
        Initialize context builder
        
        Args:
            max_context_length: Max characters cho context (để avoid token limits)
        """
        self.max_context_length = max_context_length
    
    def build_rag_context(
        self,
        question: str,
        retrieval_context: Dict,
        max_entities: int = 12,
        max_relationships: int = 20,
        include_paths: bool = True,
        include_sources: bool = True
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
                'direction': rel.get('direction', 'outgoing')
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
        include_instructions: bool = True
    ) -> str:
        """
        Args:
            context: Context từ build_rag_context()
            prompt_type: Loại prompt (qa, summary, explain, timeline)
            include_instructions: Include system instructions
            
        Returns:
            Formatted prompt string
        """
        if prompt_type == "qa":
            return self._format_qa_prompt(context, include_instructions)
        elif prompt_type == "summary":
            return self._format_summary_prompt(context, include_instructions)
        elif prompt_type == "explain":
            return self._format_explain_prompt(context, include_instructions)
        elif prompt_type == "timeline":
            return self._format_timeline_prompt(context, include_instructions)
        else:
            return self._format_qa_prompt(context, include_instructions)
    
    def _format_qa_prompt(self, context: Dict, include_instructions: bool) -> str:
        """Format prompt cho Q&A task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp.

HƯỚNG DẪN:
- Chỉ sử dụng thông tin từ context được cung cấp
- Trả lời chính xác, có căn cứ
- Nếu không có đủ thông tin, hãy nói rõ
- Sử dụng tiếng Việt tự nhiên và dễ hiểu
YÊU CẦU TRẢ LỜI:
- Trả lời theo dạng đoạn văn mạch lạc
- Nêu rõ:
  + Quy định / nội dung chính
  + Dẫn chứng từ context
- Cuối câu ghi nguồn: Ví dụ: (Nghị định 168/2024/NĐ-CP, Điều 5)

""")
        parts.append("CONTEXT:")
        parts.append("")
        
        # Entities section
        parts.append("ENTITIES:")
        for i, entity in enumerate(context['entities'], 1):
            parts.append(f"\n{i}. {entity['name']} ({entity['type']})")
            if entity['description']:
                parts.append(f"   Mô tả: {entity['description']}")
            parts.append(f"   Độ liên quan: {entity['relevance_score']:.3f}")
        
        # Relationships section
        if context['relationships']:
            parts.append("\nRELATIONSHIPS:")
            for i, rel in enumerate(context['relationships'], 1):
                arrow = "->" if rel['direction'] == 'outgoing' else "<-"
                parts.append(f"\n{i}. {rel['source']} {arrow} [{rel['relation']}] {arrow} {rel['target']}")
                if rel['description']:
                    parts.append(f"   Chi tiết: {rel['description']}")
        
        parts.append("\n" + "-" * 10)
        if context.get('paths'):
            parts.append("\nCONNECTIONS (Kết nối):")
            for i, path in enumerate(context['paths'], 1):
                route_str = " → ".join(path['route'])
                parts.append(f"\n{i}. {path['from']} đến {path['to']}: {route_str}")
        
        parts.append("\n" + "-" * 10)
        
        # Source content section
        if context.get('sources'):
            parts.append("\nSOURCE CONTENT:")
            for i, source in enumerate(context['sources'], 1):
                parts.append(f"\n{i}. {source['citation']}")
                if 'content' in source:
                    parts.append(f"{source['content'][:1500]}...")  # Limit mỗi page
        
        parts.append("\n" + "-" * 10)
        parts.append(f"\nCÂU HỎI: {context['question']}")
        
        full_prompt = "\n".join(parts)
        
        return full_prompt
    
    def _format_summary_prompt(self, context: Dict, include_instructions: bool) -> str:
        """Format prompt cho summarization task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là tóm tắt thông tin từ dữ liệu sau.

HƯỚNG DẪN:
- Tạo tóm tắt ngắn gọn, súc tích (3-5 câu)
- Bao gồm các thông tin chính: QUY ĐỊNH, MỨC PHẠT, ĐỐI TƯỢNG ÁP DỤNG
- Highlight các entities và relationships quan trọng
- Sử dụng thứ tự logic nếu có
- Viết bằng tiếng Việt tự nhiên
""")
        
        parts.append("CONTEXT:")
        parts.append("")
        
        # Compact entity list
        parts.append("KEY ENTITIES:")
        entity_by_type = {}
        for entity in context['entities']:
            etype = entity['type']
            if etype not in entity_by_type:
                entity_by_type[etype] = []
            entity_by_type[etype].append(entity['name'])
        
        for etype, names in entity_by_type.items():
            parts.append(f"  {etype}: {', '.join(names[:5])}")
        
        parts.append("")
        parts.append("KEY RELATIONSHIPS:")
        for i, rel in enumerate(context['relationships'][:10], 1):
            parts.append(f"  {i}. {rel['source']} --[{rel['relation']}]--> {rel['target']}")
        
        parts.append("\n" + "=" * 70)
        parts.append(f"\nYÊU CẦU: {context['question']}")
        parts.append("\nTÓM TẮT:")
        
        return "\n".join(parts)
    
    def _format_explain_prompt(self, context: Dict, include_instructions: bool) -> str:
        """Format prompt cho explanation task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là giải thích mối quan hệ và bối cảnh pháp lý.

HƯỚNG DẪN:
- Giải thích rõ ràng mối quan hệ giữa các entities
- Cung cấp bối cảnh pháp lý đầy đủ
- Phân tích quy định và hậu quả
- Sử dụng ví dụ cụ thể từ context
- Viết theo phong cách giáo dục, dễ hiểu
""")
        
        parts.append("CONTEXT:")
        parts.append("")
        
        # Full entities
        parts.append("ENTITIES:")
        for i, entity in enumerate(context['entities'][:8], 1):
            parts.append(f"\n{i}. {entity['name']} ({entity['type']})")
            if entity['description']:
                parts.append(f"   {entity['description']}")
        
        # Full relationships
        parts.append("\n\nRELATIONSHIPS:")
        for i, rel in enumerate(context['relationships'][:15], 1):
            parts.append(f"{i}. {rel['source']} --[{rel['relation']}]--> {rel['target']}")
            if rel['description']:
                parts.append(f"   {rel['description']}")
        
        # Paths for deeper understanding
        if context.get('paths'):
            parts.append("\n\nCONNECTIONS:")
            for path in context['paths']:
                route = " → ".join(path['route'])
                parts.append(f"- {route}")
        
        parts.append("\n" + "=" * 70)
        parts.append(f"\nCÂU HỎI: {context['question']}")
        parts.append("\nGIẢI THÍCH CHI TIẾT:")
        
        return "\n".join(parts)
    
    def _format_timeline_prompt(self, context: Dict, include_instructions: bool) -> str:
        """Format prompt cho timeline construction"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là tạo danh sách các quy định theo mức độ nghiêm trọng.

HƯỚNG DẪN:
- Sắp xếp quy định theo mức phạt hoặc mức độ vi phạm
- Ghi rõ mức phạt/hình thức xử lý cho mỗi vi phạm
- Mô tả ngắn gọn mỗi quy định
- Highlight các điểm quan trọng
- Format: Vi phạm: Mức phạt - Mô tả
""")
        
        parts.append("=" * 70)
        parts.append("CONTEXT:")
        parts.append("=" * 70)
        parts.append("")
        
        # Filter time entities
        time_entities = [e for e in context['entities'] if e['type'] == 'TIME']
        event_entities = [e for e in context['entities'] if e['type'] == 'EVENT']
        
        if time_entities:
            parts.append("THỜI GIAN:")
            for entity in time_entities:
                parts.append(f"  - {entity['name']}: {entity['description']}")
        
        parts.append("\nSỰ KIỆN:")
        for entity in event_entities:
            parts.append(f"  - {entity['name']}: {entity['description']}")
        
        parts.append("\nQUAN HỆ:")
        for rel in context['relationships'][:15]:
            parts.append(f"  - {rel['source']} → {rel['target']} ({rel['relation']})")
        
        parts.append("\n" + "=" * 70)
        parts.append(f"\nYÊU CẦU: {context['question']}")
        parts.append("\nTIMELINE:")
        
        return "\n".join(parts)
    
    def create_multi_turn_context(
        self,
        conversation_history: List[Dict],
        current_context: Dict,
        max_history: int = 3
    ) -> str:
        """
        Tạo context cho multi-turn conversation
        
        Args:
            conversation_history: List of {question, answer} dicts
            current_context: Current retrieval context
            max_history: Max previous turns to include
            
        Returns:
            Formatted context với history
        """
        parts = []
        
        parts.append("CONVERSATION HISTORY:")
        parts.append("=" * 70)
        
        # Include recent history
        recent_history = conversation_history[-max_history:] if conversation_history else []
        
        for i, turn in enumerate(recent_history, 1):
            parts.append(f"\nTurn {i}:")
            parts.append(f"Q: {turn['question']}")
            parts.append(f"A: {turn['answer'][:200]}...")  # Truncate long answers
        
        if not recent_history:
            parts.append("(First question)")
        
        parts.append("\n" + "=" * 70)
        parts.append("CURRENT CONTEXT:")
        parts.append("=" * 70)
        
        # Add current context
        parts.append(self.format_for_gemini(current_context, prompt_type="qa", include_instructions=False))
        
        return "\n".join(parts)
    
    def estimate_token_count(self, text: str) -> int:
        """
        Ước tính token count (rough estimate)
        Vietnamese: ~1 token per 3-4 characters
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 3.5 characters for Vietnamese
        return len(text) // 4
    
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
            formatted = self.format_for_gemini(truncated, prompt_type="qa")
            estimated_tokens = self.estimate_token_count(formatted)
            
            if estimated_tokens <= max_tokens:
                break
            
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
    
    def add_examples_to_prompt(
        self,
        base_prompt: str,
        examples: List[Dict],
        max_examples: int = 2
    ) -> str:
        """
        Add few-shot examples to prompt
        
        Args:
            base_prompt: Base prompt
            examples: List of {question, answer} examples
            max_examples: Max examples to include
            
        Returns:
            Prompt với examples
        """
        if not examples:
            return base_prompt
        
        parts = [base_prompt]
        parts.append("VÍ DỤ:")
        for i, example in enumerate(examples[:max_examples], 1):
            parts.append(f"\nVí dụ {i}:")
            parts.append(f"Câu hỏi: {example['question']}")
            parts.append(f"Trả lời: {example['answer']}")
        
        parts.append("BÂY GIỜ HÃY TRẢ LỜI CÂU HỎI HIỆN TẠI THEO CÁCH TƯƠNG TỰ:")
        
        return "\n".join(parts)