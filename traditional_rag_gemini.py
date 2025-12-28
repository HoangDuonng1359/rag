"""
Traditional RAG - Gemini Integration
Complete pipeline: Retrieval → Context Building → LLM Generation
"""

import os
from typing import Dict, Optional
import time
import google.generativeai as genai
from dotenv import load_dotenv

from traditional_rag_retriever import TraditionalRAGRetriever
from traditional_rag_context import TraditionalContextBuilder

load_dotenv()


class TraditionalGeminiRAG:
    """
    Traditional RAG system với Google Gemini API
    
    Pipeline: Query → Retrieval → Context Building → LLM Generation
    """
    
    def __init__(
        self,
        retriever: Optional[TraditionalRAGRetriever] = None,
        context_builder: Optional[TraditionalContextBuilder] = None,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        """
        Initialize Traditional RAG với Gemini
        
        Args:
            retriever: TraditionalRAGRetriever instance (nếu None sẽ tạo mới)
            context_builder: TraditionalContextBuilder instance (nếu None sẽ tạo mới)
            api_key: Google API key
            model_name: Gemini model name
        """
        # Initialize retriever
        if retriever is None:
            print("Initializing new retriever...")
            self.retriever = TraditionalRAGRetriever()
        else:
            self.retriever = retriever
        
        # Initialize context builder
        if context_builder is None:
            self.context_builder = TraditionalContextBuilder(max_context_length=8000)
        else:
            self.context_builder = context_builder
        
        # Initialize Gemini
        self.model_name = model_name
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        
        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Safety settings (permissive cho educational content)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        
        print(f"✓ Traditional RAG initialized with {model_name}")
    
    def answer(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: Optional[float] = None,
        prompt_type: str = "qa",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        stream: bool = False
    ) -> Dict:
        """
        Generate answer cho question
        
        Args:
            question: User question
            top_k: Số chunks retrieve
            min_similarity: Minimum similarity threshold
            prompt_type: 'qa', 'summary', 'explain'
            temperature: Generation temperature
            max_tokens: Max output tokens
            stream: Stream response
            
        Returns:
            Dict với answer, context, và metrics
        """
        start_time = time.time()
        
        # Step 1: Retrieve
        print(f"\n1. Retrieving top-{top_k} chunks...")
        retrieval_start = time.time()
        
        chunks = self.retriever.retrieve(
            question,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        retrieval_time = time.time() - retrieval_start
        print(f"   Retrieved {len(chunks)} chunks in {retrieval_time:.2f}s")
        
        if not chunks:
            return {
                'question': question,
                'answer': "Không tìm thấy thông tin liên quan trong dữ liệu.",
                'chunks': [],
                'retrieval_time': retrieval_time,
                'generation_time': 0.0,
                'total_time': time.time() - start_time
            }
        
        # Step 2: Build context
        print(f"2. Building context...")
        context_start = time.time()
        
        prompt = self.context_builder.format_for_gemini(
            question,
            chunks,
            max_chunks=top_k,
            prompt_type=prompt_type
        )
        
        context_time = time.time() - context_start
        print(f"   Context built in {context_time:.2f}s")
        print(f"   Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
        
        # Step 3: Generate answer
        print(f"3. Generating answer with Gemini...")
        generation_start = time.time()
        
        try:
            if stream:
                # Streaming response
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    safety_settings=self.safety_settings,
                    stream=True
                )
                
                answer_parts = []
                for chunk in response:
                    if chunk.text:
                        answer_parts.append(chunk.text)
                        print(chunk.text, end='', flush=True)
                
                answer = ''.join(answer_parts)
                print()  # Newline after streaming
                
            else:
                # Non-streaming response
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                    safety_settings=self.safety_settings
                )
                
                answer = response.text
            
            generation_time = time.time() - generation_start
            
            # Check finish reason
            try:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == 2:  # MAX_TOKENS
                    print(f"   Warning: Response truncated (MAX_TOKENS reached)")
            except:
                pass
            
            print(f"   Generated in {generation_time:.2f}s")
            
            # Get token usage
            try:
                usage = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
                print(f"   Tokens: {usage['prompt_tokens']} + {usage['completion_tokens']} = {usage['total_tokens']}")
            except:
                usage = None
            
        except Exception as e:
            print(f"   Error during generation: {e}")
            answer = f"Lỗi khi generate answer: {str(e)}"
            generation_time = time.time() - generation_start
            usage = None
        
        total_time = time.time() - start_time
        
        # Build result
        result = {
            'question': question,
            'answer': answer,
            'chunks': chunks,
            'chunk_count': len(chunks),
            'retrieval_time': retrieval_time,
            'context_time': context_time,
            'generation_time': generation_time,
            'total_time': total_time,
            'usage': usage,
            'model': self.model_name
        }
        
        print(f"\n✓ Total time: {total_time:.2f}s")
        
        return result
    
    def batch_answer(
        self,
        questions: list[str],
        top_k: int = 5,
        prompt_type: str = "qa"
    ) -> list[Dict]:
        """
        Answer multiple questions
        
        Args:
            questions: List of questions
            top_k: Chunks per question
            prompt_type: Prompt type
            
        Returns:
            List of results
        """
        results = []
        
        print(f"\n=== Batch Processing {len(questions)} questions ===")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{'='*80}")
            print(f"Question {i}/{len(questions)}")
            print(f"{'='*80}")
            
            result = self.answer(
                question,
                top_k=top_k,
                prompt_type=prompt_type,
                stream=False
            )
            
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get system metrics"""
        retriever_metrics = self.retriever.get_metrics()
        
        return {
            'retriever': retriever_metrics,
            'model': self.model_name,
            'chunks_in_index': retriever_metrics['chunks_in_index']
        }


# Example usage và tests
if __name__ == "__main__":
    print("=== Initializing Traditional RAG với Gemini ===")
    
    rag = TraditionalGeminiRAG()
    
    # Test single question
    print("\n" + "="*80)
    print("=== TEST QUESTION ===")
    print("="*80)
    
    question = "Mức phạt không đội mũ bảo hiểm là bao nhiêu?"
    
    result = rag.answer(
        question,
        top_k=5,
        min_similarity=0.5,
        prompt_type="qa",
        stream=False
    )
    
    print("\n" + "="*80)
    print("=== ANSWER ===")
    print("="*80)
    print(result['answer'])
    
    print("\n" + "="*80)
    print("=== RETRIEVED CHUNKS ===")
    print("="*80)
    for i, chunk in enumerate(result['chunks'], 1):
        print(f"{i}. {chunk['id']} (similarity: {chunk['similarity']:.3f})")
        print(f"   Pages: {chunk['metadata']['page_start']}-{chunk['metadata']['page_end']}")
    
    print("\n" + "="*80)
    print("=== METRICS ===")
    print("="*80)
    print(f"Retrieval: {result['retrieval_time']:.2f}s")
    print(f"Context: {result['context_time']:.2f}s")
    print(f"Generation: {result['generation_time']:.2f}s")
    print(f"Total: {result['total_time']:.2f}s")
    
    if result['usage']:
        print(f"\nTokens:")
        print(f"  Prompt: {result['usage']['prompt_tokens']}")
        print(f"  Completion: {result['usage']['completion_tokens']}")
        print(f"  Total: {result['usage']['total_tokens']}")
