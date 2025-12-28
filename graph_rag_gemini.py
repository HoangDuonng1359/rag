import os
from typing import List, Dict, Optional, Generator
import time
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class GeminiRAG:
    """
    Graph RAG integration với Google Gemini API
    """
    
    def __init__(
        self,
        hybrid_retriever,
        context_builder,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash"
    ):
        """
        
        Args:
            hybrid_retriever: HybridRetriever instance
            context_builder: ContextBuilder instance
            api_key: Google API key 
            model_name: Gemini model name 
        """
        self.retriever = hybrid_retriever
        self.builder = context_builder
        self.model_name = model_name
        api_key = api_key or os.getenv('GOOGLE_API_KEY')

        # Configure API key for the latest SDK
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Generation config
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,  # Tăng lên 8192 để đủ cho câu trả lời chi tiết
        }
        
        # Safety settings (permissive cho educational content)
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        print(f"Gemini initialized with {model_name}")
    
    def generate_answer(
        self,
        question: str,
        prompt_type: str = "qa",
        retrieval_params: Optional[Dict] = None,
        context_params: Optional[Dict] = None,
        temperature: float = 0.7,
        max_tokens: int = 8192  # Tăng lên 8192 để đủ cho câu trả lời đầy đủ
    ) -> Dict:
        """
        Generate answer cho question sử dụng RAG pipeline
        
        Args:
            question: User question
            prompt_type: Prompt type (qa, summary, explain, timeline)
            retrieval_params: Parameters cho hybrid retrieval
            context_params: Parameters cho context building
            temperature: Generation temperature
            max_tokens: Max output tokens
            
        Returns:
            Dict với answer, context, và metadata
        """
        start_time = time.time()
        
        # Default parameters
        retrieval_params = retrieval_params or {
            'top_k': 10,
            'vector_top_k': 5,
            'expansion_depth': 1
        }
        
        context_params = context_params or {
            'max_entities': 10,
            'max_relationships': 15,
            'include_paths': False
        }
        
        # Retrieve context
        retrieval_context = self.retriever.retrieve(
            question=question,
            **retrieval_params
        )
        retrieval_time = time.time() - start_time
        # Build RAG context
        rag_context = self.builder.build_rag_context(
            question=question,
            retrieval_context=retrieval_context,
            **context_params
        )
        
        #Format prompt
        prompt = self.builder.format_for_gemini(
            context=rag_context,
            prompt_type=prompt_type,
            include_instructions=True
        )
        print(prompt)
        # Update generation config
        gen_config = self.generation_config.copy()
        gen_config['temperature'] = temperature
        gen_config['max_output_tokens'] = max_tokens
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=gen_config,
                safety_settings=self.safety_settings
            )
            
            # Check if response was truncated
            answer = response.text
            
            # Check finish reason
            finish_reason = None
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'finish_reason'):
                    finish_reason = candidate.finish_reason
                    if finish_reason != 1:  # 1 = STOP (normal completion)
                        print(f"Warning: Response may be incomplete. Finish reason: {finish_reason}")
                        print(f"     (1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER)")
            
            generation_time = time.time() - start_time - retrieval_time
            print(f"Generated answer in {generation_time:.2f}s")
            print(f"Answer length: {len(answer)} chars")
            
            usage_metadata = {}
            if hasattr(response, 'usage_metadata'):
                usage_metadata = {
                    'prompt_tokens': response.usage_metadata.prompt_token_count,
                    'completion_tokens': response.usage_metadata.candidates_token_count,
                    'total_tokens': response.usage_metadata.total_token_count
                }
                print(f"Token usage: {usage_metadata}")
                
            if finish_reason:
                usage_metadata['finish_reason'] = finish_reason
        except Exception as e:
            print(f"Generation error: {e}")
            answer = f"Error generating answer: {str(e)}"
            usage_metadata = {}
        
        total_time = time.time() - start_time
        
        # Build result
        result = {
            'question': question,
            'answer': answer,
            'context': rag_context,
            'prompt_type': prompt_type,
            'metadata': {
                'retrieval_time': retrieval_time,
                'generation_time': total_time - retrieval_time,
                'total_time': total_time,
                'model': self.model_name,
                'entities_used': len(rag_context['entities']),
                'relationships_used': len(rag_context['relationships']),
                **usage_metadata
            }
        }
        
        print(f"Total time: {total_time:.2f}s\n")
        
        return result
    
    def generate_answer_stream(
        self,
        question: str,
        prompt_type: str = "qa",
        retrieval_params: Optional[Dict] = None,
        context_params: Optional[Dict] = None,
        temperature: float = 0.7
    ) -> Generator[str, None, Dict]:
        """
        Generate answer với streaming (real-time token generation)
        
        Args:
            question: User question
            prompt_type: Prompt type
            retrieval_params: Retrieval parameters
            context_params: Context parameters
            temperature: Generation temperature
            
        Yields:
            Answer tokens as they're generated
            
        Returns:
            Final result dict với complete answer
        """
        start_time = time.time()
        
        # Default parameters
        retrieval_params = retrieval_params or {
            'top_k': 10,
            'vector_top_k': 5,
            'expansion_depth': 1
        }
        
        context_params = context_params or {
            'max_entities': 10,
            'max_relationships': 15,
            'include_paths': False
        }
        
        # Retrieve context (không stream phần này)
        retrieval_context = self.retriever.retrieve(
            question=question,
            **retrieval_params
        )
        retrieval_time = time.time() - start_time
        
        # Build RAG context
        rag_context = self.builder.build_rag_context(
            question=question,
            retrieval_context=retrieval_context,
            **context_params
        )
        
        # Format prompt
        prompt = self.builder.format_for_gemini(
            context=rag_context,
            prompt_type=prompt_type,
            include_instructions=True
        )
        
        # Update generation config
        gen_config = self.generation_config.copy()
        gen_config['temperature'] = temperature
        
        # Generate với streaming
        full_answer = ""
        
        try:
            # Streaming with latest SDK
            for chunk in self.model.generate_content(
                prompt,
                generation_config=gen_config,
                safety_settings=self.safety_settings,
                stream=True
            ):
                text = getattr(chunk, 'text', None)
                if text:
                    full_answer += text
                    yield text
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            full_answer = error_msg
            yield error_msg
        
        total_time = time.time() - start_time
        
        # Return final result
        result = {
            'question': question,
            'answer': full_answer,
            'context': rag_context,
            'prompt_type': prompt_type,
            'metadata': {
                'retrieval_time': retrieval_time,
                'generation_time': total_time - retrieval_time,
                'total_time': total_time,
                'model': self.model_name,
                'entities_used': len(rag_context['entities']),
                'relationships_used': len(rag_context['relationships'])
            }
        }
        
        return result
    
    def chat(
        self,
        conversation_history: List[Dict],
        question: str,
        max_history: int = 3
    ) -> Dict:
        """
        Multi-turn conversation với context awareness
        
        Args:
            conversation_history: List of previous {question, answer} pairs
            question: Current question
            max_history: Max previous turns to include
            
        Returns:
            Result dict với answer
        """
        # Retrieve context
        retrieval_context = self.retriever.retrieve(
            question=question,
            top_k=10,
            vector_top_k=5
        )
        
        # Build context
        rag_context = self.builder.build_rag_context(
            question=question,
            retrieval_context=retrieval_context
        )
        
        # Create multi-turn prompt
        prompt = self.builder.create_multi_turn_context(
            conversation_history=conversation_history,
            current_context=rag_context,
            max_history=max_history
        )
        
        # Generate
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            answer = response.text
        except Exception as e:
            answer = f"Error: {str(e)}"
        
        return {
            'question': question,
            'answer': answer,
            'context': rag_context
        }
    
    def batch_generate(
        self,
        questions: List[str],
        prompt_type: str = "qa"
    ) -> List[Dict]:
        """
        Generate answers cho multiple questions
        
        Args:
            questions: List of questions
            prompt_type: Prompt type
            
        Returns:
            List of result dicts
        """
        results = []
        
        print(f"Processing {len(questions)} questions...\n")
        
        for i, question in enumerate(questions, 1):
            print(f"[{i}/{len(questions)}] {question[:60]}...")
            
            result = self.generate_answer(
                question=question,
                prompt_type=prompt_type
            )
            
            results.append(result)
        
        print(f"\nCompleted {len(results)} questions")
        
        return results
    
    def compare_prompt_types(
        self,
        question: str
    ) -> Dict[str, Dict]:
        """
        Compare different prompt types cho same question
        
        Args:
            question: Question to test
            
        Returns:
            Dict mapping prompt_type → result
        """
        prompt_types = ["qa", "summary", "explain"]
        results = {}
        
        print(f"Comparing prompt types for: {question}\n")
        
        for ptype in prompt_types:
            print(f"Testing {ptype}...")
            result = self.generate_answer(
                question=question,
                prompt_type=ptype,
                temperature=0.7
            )
            results[ptype] = result
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information về Gemini model being used"""
        try:
            model_info = genai.get_model(f"models/{self.model_name}")
            return {
                'name': getattr(model_info, 'name', None),
                'description': getattr(model_info, 'description', None),
                'input_token_limit': getattr(model_info, 'input_token_limit', None),
                'output_token_limit': getattr(model_info, 'output_token_limit', None),
                'supported_generation_methods': getattr(model_info, 'supported_generation_methods', None)
            }
        except Exception as e:
            return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    from langchain_community.graphs import Neo4jGraph
    from graph_rag_query import GraphRAGQuery
    from graph_rag_embeddings import EntityEmbeddings
    from graph_rag_hybrid import HybridRetriever
    from graph_rag_context import ContextBuilder
    
    # Setup
    NEO4J_URI="neo4j+s://41ab799a.databases.neo4j.io"
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="xmriUzmvo9dSAyc10u9mpB7nzyQHMZFooKqH5yBP2d4"
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Initialize 
    graph_query = GraphRAGQuery(graph)
    embeddings = EntityEmbeddings(graph)
    hybrid = HybridRetriever(graph_query, embeddings)
    builder = ContextBuilder()
    
    gemini_rag = GeminiRAG(
        hybrid_retriever=hybrid,
        context_builder=builder,
        model_name="gemini-2.5-flash", 
        api_key=os.getenv('GOOGLE_API_KEY')
    )
    # Test
    result = gemini_rag.generate_answer(
        question="Mức phạt vi phạm vượt đèn đỏ đối với xe máy là bao nhiêu?",
        prompt_type="qa",
        max_tokens=8192
    )
    
    print("\n=== ANSWER ===")
    print(result['answer'])
    
    print("\n=== METADATA ===")
    print(f"Entities used: {result['metadata']['entities_used']}")
    print(f"Relationships used: {result['metadata']['relationships_used']}")
    print(f"Total time: {result['metadata']['total_time']:.2f}s")
    
    # Test với câu hỏi phức tạp hơn
    print("\n" + "="*70)
    print("=== TEST 2: Câu hỏi phức tạp ===")
    print("="*70)
    
    result2 = gemini_rag.generate_answer(
        question="So sánh mức phạt không đội mũ bảo hiểm giữa xe máy và ô tô",
        prompt_type="explain",
        max_tokens=8192
    )
    
    print("\n=== ANSWER ===")
    print(result2['answer'])
