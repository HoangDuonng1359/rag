"""
Traditional RAG - Context Builder
Format retrieved chunks thành structured prompts cho Gemini
"""

from typing import List, Dict, Any, Optional


class TraditionalContextBuilder:
    """
    Build context từ retrieved chunks thành prompts cho LLM
    
    Đơn giản hơn Graph RAG - chỉ cần format chunks và metadata
    """
    
    def __init__(self, max_context_length: int = 8000):
        """
        Initialize context builder
        
        Args:
            max_context_length: Max characters cho context (avoid token limits)
        """
        self.max_context_length = max_context_length
    
    def build_context(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Build context từ retrieved chunks
        
        Args:
            question: User question
            retrieved_chunks: Results từ retriever
            max_chunks: Max số chunks to include
            
        Returns:
            Dict với structured context
        """
        # Limit chunks
        if max_chunks is not None:
            chunks = retrieved_chunks[:max_chunks]
        else:
            chunks = retrieved_chunks
        
        # Extract và deduplicate sources
        sources = self._extract_sources(chunks)
        
        # Build context
        context = {
            'question': question,
            'chunks': self._format_chunks(chunks),
            'sources': sources,
            'chunk_count': len(chunks),
            'total_words': sum(c['metadata'].get('word_count', 0) for c in chunks),
            'page_range': self._get_page_range(chunks)
        }
        
        return context
    
    def _format_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format chunks with metadata"""
        formatted = []
        
        for i, chunk in enumerate(chunks, 1):
            formatted.append({
                'number': i,
                'id': chunk['id'],
                'content': chunk['content'],
                'similarity': round(chunk['similarity'], 4),
                'pages': f"{chunk['metadata']['page_start']}-{chunk['metadata']['page_end']}",
                'word_count': chunk['metadata'].get('word_count', 0),
                'chapter': chunk['metadata'].get('chapter', 'N/A')
            })
        
        return formatted
    
    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Extract unique page citations từ chunks"""
        pages = set()
        
        for chunk in chunks:
            start = chunk['metadata']['page_start']
            end = chunk['metadata']['page_end']
            
            if start == end:
                pages.add(f"Trang {start}")
            else:
                pages.add(f"Trang {start}-{end}")
        
        return sorted(list(pages))
    
    def _get_page_range(self, chunks: List[Dict[str, Any]]) -> str:
        """Get overall page range covered"""
        if not chunks:
            return "N/A"
        
        all_pages = []
        for chunk in chunks:
            all_pages.append(chunk['metadata']['page_start'])
            all_pages.append(chunk['metadata']['page_end'])
        
        return f"{min(all_pages)}-{max(all_pages)}"
    
    def format_prompt(
        self,
        context: Dict[str, Any],
        prompt_type: str = "qa",
        include_instructions: bool = True
    ) -> str:
        """
        Format context thành prompt string cho Gemini
        
        Args:
            context: Context từ build_context()
            prompt_type: 'qa', 'summary', 'explain'
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
        else:
            return self._format_qa_prompt(context, include_instructions)
    
    def _format_qa_prompt(
        self,
        context: Dict[str, Any],
        include_instructions: bool
    ) -> str:
        """Format prompt cho Q&A task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các đoạn văn được cung cấp.

HƯỚNG DẪN:
- Chỉ sử dụng thông tin từ các đoạn văn được cung cấp
- Trả lời chính xác, chi tiết và có căn cứ
- Nếu không có đủ thông tin, hãy nói rõ
- Trích dẫn số thứ tự đoạn văn khi trả lời (ví dụ: "Theo đoạn 1...")
- Đưa ra trang nguồn khi có thể
- Sử dụng tiếng Việt tự nhiên và dễ hiểu
""")
        
        parts.append("CONTEXT:")
        parts.append("")
        
        # Chunks section
        parts.append(f"CÁC ĐOẠN VĂN LIÊN QUAN ({context['chunk_count']} đoạn):")
        parts.append("")
        
        for chunk in context['chunks']:
            parts.append(f"--- Đoạn {chunk['number']} ---")
            parts.append(f"Nguồn: {chunk['pages']}")
            parts.append(f"Độ liên quan: {chunk['similarity']:.3f}")
            parts.append("")
            parts.append(chunk['content'])
            parts.append("")
        
        parts.append("-" * 80)
        
        # Sources section
        if context['sources']:
            parts.append("\nTÀI LIỆU THAM KHẢO:")
            for source in context['sources']:
                parts.append(f"- {source}")
        
        parts.append("\n" + "-" * 80)
        parts.append(f"\nCÂU HỎI: {context['question']}")
        parts.append("\nTRẢ LỜI:")
        
        full_prompt = "\n".join(parts)
        
        # Check length
        if len(full_prompt) > self.max_context_length:
            # Truncate chunks if too long
            print(f"Warning: Prompt length ({len(full_prompt)}) exceeds max ({self.max_context_length})")
            # TODO: Implement smart truncation
        
        return full_prompt
    
    def _format_summary_prompt(
        self,
        context: Dict[str, Any],
        include_instructions: bool
    ) -> str:
        """Format prompt cho summarization"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là tóm tắt các đoạn văn sau.

HƯỚNG DẪN:
- Tạo tóm tắt ngắn gọn, súc tích (3-5 câu)
- Bao gồm các thông tin chính
- Highlight các quy định và điều khoản quan trọng
- Sử dụng thứ tự logic nếu có
""")
        
        parts.append("CÁC ĐOẠN VĂN CẦN TÓM TẮT:")
        parts.append("")
        
        for chunk in context['chunks']:
            parts.append(f"Đoạn {chunk['number']}:")
            parts.append(chunk['content'])
            parts.append("")
        
        parts.append("TÓM TẮT:")
        
        return "\n".join(parts)
    
    def _format_explain_prompt(
        self,
        context: Dict[str, Any],
        include_instructions: bool
    ) -> str:
        """Format prompt cho explanation task"""
        parts = []
        
        if include_instructions:
            parts.append("""Bạn là một giảng viên luật giao thông Việt Nam. Nhiệm vụ của bạn là giải thích về chủ đề được hỏi.

HƯỚNG DẪN:
- Giải thích rõ ràng, dễ hiểu
- Bao gồm quy định, lý do, hậu quả
- Sử dụng examples từ context
- Giải thích các thuật ngữ pháp lý
""")
        
        parts.append("THÔNG TIN:")
        parts.append("")
        
        for chunk in context['chunks']:
            parts.append(chunk['content'])
            parts.append("")
        
        parts.append(f"CÂU HỎI: {context['question']}")
        parts.append("\nGIẢI THÍCH:")
        
        return "\n".join(parts)
    
    def format_for_gemini(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_chunks: int = 5,
        prompt_type: str = "qa"
    ) -> str:
        """
        Convenience method: build context + format prompt trong một call
        
        Args:
            question: User question
            retrieved_chunks: Retrieved chunks
            max_chunks: Max chunks to include
            prompt_type: Prompt type
            
        Returns:
            Formatted prompt string
        """
        context = self.build_context(question, retrieved_chunks, max_chunks)
        prompt = self.format_prompt(context, prompt_type, include_instructions=True)
        return prompt


# Example usage và tests
if __name__ == "__main__":
    # Mock retrieved chunks
    mock_chunks = [
        {
            'id': 'chunk_0009',
            'content': 'Cách mạng tháng Tám 1945 đã thành công, nước Việt Nam dân chủ cộng hòa đã được thành lập. Đây là thắng lợi vĩ đại của nhân dân Việt Nam dưới sự lãnh đạo của Đảng Cộng sản Đông Dương và Chủ tịch Hồ Chí Minh.',
            'similarity': 0.8523,
            'metadata': {
                'page_start': 14,
                'page_end': 22,
                'word_count': 50,
                'chapter': 'Chương I'
            }
        },
        {
            'id': 'chunk_0015',
            'content': 'Nguyên nhân thắng lợi của Cách mạng tháng Tám bao gồm: sự lãnh đạo đúng đắn của Đảng, lực lượng dân tộc đoàn kết rộng rãi trong Mặt trận Việt Minh, thời cơ cách mạng thuận lợi khi Nhật đầu hàng.',
            'similarity': 0.8102,
            'metadata': {
                'page_start': 25,
                'page_end': 28,
                'word_count': 45,
                'chapter': 'Chương I'
            }
        },
        {
            'id': 'chunk_0020',
            'content': 'Sau thành công của Cách mạng tháng Tám, Chủ tịch Hồ Chí Minh đọc Tuyên ngôn Độc lập tại Quảng trường Ba Đình, Hà Nội ngày 2/9/1945, khai sinh nước Việt Nam Dân chủ Cộng hòa.',
            'similarity': 0.7856,
            'metadata': {
                'page_start': 30,
                'page_end': 32,
                'word_count': 42,
                'chapter': 'Chương I'
            }
        }
    ]
    
    print("=== Test Context Builder ===")
    
    # Initialize
    context_builder = TraditionalContextBuilder(max_context_length=8000)
    
    # Build context
    question = "Cách mạng tháng Tám năm 1945 thành công do đâu?"
    context = context_builder.build_context(question, mock_chunks)
    
    print(f"\n=== CONTEXT ===")
    print(f"Question: {context['question']}")
    print(f"Chunks: {context['chunk_count']}")
    print(f"Total words: {context['total_words']}")
    print(f"Page range: {context['page_range']}")
    print(f"Sources: {context['sources']}")
    
    # Test Q&A prompt
    print("\n" + "="*80)
    print("=== Q&A PROMPT ===")
    print("="*80)
    
    qa_prompt = context_builder.format_prompt(context, prompt_type="qa")
    print(qa_prompt)
    
    # Test prompt length
    print(f"\n=== STATS ===")
    print(f"Prompt length: {len(qa_prompt)} characters")
    print(f"Estimated tokens: ~{len(qa_prompt) // 4}")
    
    # Test convenience method
    print("\n" + "="*80)
    print("=== CONVENIENCE METHOD ===")
    print("="*80)
    
    full_prompt = context_builder.format_for_gemini(
        question,
        mock_chunks,
        max_chunks=2,
        prompt_type="qa"
    )
    
    print(f"Prompt with 2 chunks: {len(full_prompt)} chars")
