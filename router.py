
import os
from typing import Dict, Optional, Literal
from enum import Enum
from typing_extensions import TypedDict
import google.generativeai as genai
from dotenv import load_dotenv

from traditional_rag_gemini import TraditionalGeminiRAG
from graph_rag_gemini import GeminiRAG
from traditional_rag_retriever import TraditionalRAGRetriever
from traditional_rag_context import TraditionalContextBuilder

load_dotenv()


# Response schema cho classification
class ClassificationResponse(TypedDict):
    query_type: Literal["FACTUAL", "RELATIONAL"]


class QueryType(Enum):
    """Loại câu hỏi"""
    FACTUAL = "factual"  # Câu hỏi cụ thể, sự kiện
    RELATIONAL = "relational"  # Câu hỏi về mối quan hệ, logic


class IntelligentRAGRouter:
    
    def __init__(
        self,
        traditional_rag: Optional[TraditionalGeminiRAG] = None,
        graph_rag: Optional[GeminiRAG] = None,
        api_key: Optional[str] = None,
        classifier_model: str = "gemini-2.5-flash"
    ):
        self.traditional_rag = traditional_rag
        self.graph_rag = graph_rag
        
        api_key = api_key or os.getenv('GOOGLE_API_KEY')
        genai.configure(api_key=api_key)
        self.classifier_model = genai.GenerativeModel(
            classifier_model,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": ClassificationResponse
            }
        )
        
        # Prompt template cho classification
        self.classification_prompt = """Phân loại câu hỏi về luật giao thông Việt Nam:

FACTUAL - Hỏi về quy định, mức phạt, điều luật, biển báo CỤ THỂ:
- "Mức phạt vượt đèn đỏ là bao nhiêu?"
- "Biển báo P.102 có ý nghĩa gì?"
- "Quy định tốc độ tối đa trong khu dân cư?"

RELATIONAL - So sánh, mối quan hệ giữa các quy định, phân tích trường hợp, lý do:
- "So sánh mức phạt giữa xe máy và ô tô"
- "Mối liên hệ giữa nồng độ cồn và mức phạt"
- "Tại sao phải tuân thủ quy tắc giao thông?"
- "Phân tích trường hợp vi phạm X"

Câu hỏi: "{question}"

Trả về JSON với field "query_type" là "FACTUAL" hoặc "RELATIONAL"."""
    
    def classify_query(self, question: str) -> QueryType:
        """
        Phân loại câu hỏi bằng LLM với JSON schema
        
        Args:
            question: Câu hỏi cần phân loại
            
        Returns:
            QueryType: FACTUAL hoặc RELATIONAL
        """
        # Chỉ dùng LLM với JSON schema để phân loại
        try:
            prompt = self.classification_prompt.format(question=question)
            
            print(f"  → Đang phân loại bằng LLM...")
            response = self.classifier_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                }
            )
            
            # Parse JSON response
            import json
            result = json.loads(response.text)
            classification = result.get("query_type", "FACTUAL")
            
            print(f"  → LLM classification: {classification}")
            
            if classification == "RELATIONAL":
                return QueryType.RELATIONAL
            else:
                return QueryType.FACTUAL
                
        except Exception as e:
            print(f"  → Lỗi khi phân loại: {e}")
            print(f"  → Sử dụng mặc định: FACTUAL")
            return QueryType.FACTUAL
    
    def route_query(
        self,
        question: str,
        auto_classify: bool = True,
        force_type: Optional[QueryType] = None,
        **kwargs
    ) -> Dict:
        """
        Route câu hỏi đến hệ thống RAG phù hợp
        
        Args:
            question: Câu hỏi cần trả lời
            auto_classify: Tự động phân loại (True) hoặc dùng force_type
            force_type: Buộc sử dụng một loại RAG cụ thể
            **kwargs: Tham số bổ sung cho các hệ thống RAG
            
        Returns:
            Dict chứa:
                - question: Câu hỏi gốc
                - query_type: Loại câu hỏi
                - system_used: Hệ thống được sử dụng
                - answer: Câu trả lời
                - context: Context được sử dụng (nếu có)
                - metadata: Thông tin bổ sung
        """
        # Phân loại câu hỏi
        if force_type:
            query_type = force_type
            print(f"Forced query type: {query_type.value}")
        elif auto_classify:
            print("Đang phân loại câu hỏi...")
            query_type = self.classify_query(question)
            print(f"Query type: {query_type.value}")
        else:
            query_type = QueryType.FACTUAL
        
        # Route đến hệ thống phù hợp
        try:
            if query_type == QueryType.FACTUAL:
                return self._use_traditional_rag(question, **kwargs)
            else:
                return self._use_graph_rag(question, **kwargs)
        except Exception as e:
            return {
                "question": question,
                "query_type": query_type.value,
                "system_used": "error",
                "answer": f"Lỗi khi xử lý: {str(e)}",
                "context": None,
                "metadata": {"error": str(e)}
            }
    
    def _use_traditional_rag(self, question: str, **kwargs) -> Dict:
        if self.traditional_rag is None:
            raise ValueError("Traditional RAG chưa được khởi tạo")
        result = self.traditional_rag.answer(
            question=question,
            top_k=kwargs.get('top_k', 10),
            stream=kwargs.get('stream', False)
        )
        
        return {
            "question": question,
            "query_type": QueryType.FACTUAL.value,
            "system_used": "traditional_rag",
            "answer": result.get("answer", ""),
            "context": "", 
            "metadata": {
                "chunks": result.get("chunks", []),
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "total_time": result.get("total_time", 0)
            }
        }
    
    def _use_graph_rag(self, question: str, **kwargs) -> Dict:
        if self.graph_rag is None:
            raise ValueError("Graph RAG chưa được khởi tạo")
        
        print("  → Đang sử dụng Graph RAG...")
        
        # Query Graph RAG
        retrieval_params = {
            'top_k': kwargs.get('top_k', 10),
            'vector_top_k': kwargs.get('top_k_vector', 5),
            'expansion_depth': kwargs.get('expansion_depth', 1)
        }
        
        print(f"  → Đang truy vấn Neo4j graph database...")
        print(f"  → Retrieval params: {retrieval_params}")
        
        try:
            result = self.graph_rag.generate_answer(
                question=question,
                retrieval_params=retrieval_params,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 8192)
            )
            print(f"  → Graph RAG hoàn tất!")
        except Exception as e:
            print(f"  → Lỗi khi gọi Graph RAG: {e}")
            raise
        
        return {
            "question": question,
            "query_type": QueryType.RELATIONAL.value,
            "system_used": "graph_rag",
            "answer": result.get("answer", ""),
            "context": result.get("context", ""),
            "metadata": {
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "total_time": result.get("total_time", 0)
            }
        }
    

def create_router_with_defaults(
    chroma_path: str = "chroma_db",
    api_key: Optional[str] = None
) -> IntelligentRAGRouter:
    
    # Khởi tạo Traditional RAG
    print("\nKhởi tạo Traditional RAG...")
    traditional_rag = TraditionalGeminiRAG(
        api_key=api_key,
        model_name="gemini-2.5-flash"
    )
    
    print("\nKhởi tạo Graph RAG...")
    try:
        try:
            from langchain_neo4j import Neo4jGraph
        except ImportError:
            from langchain_community.graphs import Neo4jGraph
        
        from graph_rag_query import GraphRAGQuery
        from graph_rag_embeddings import EntityEmbeddings
        from graph_rag_hybrid import HybridRetriever
        from graph_rag_context import ContextBuilder
        
        # Khởi tạo Neo4j connection
        neo4j_graph = Neo4jGraph(
            url=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            username=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD', '')
        )
        
        # Khởi tạo Graph Query và Embeddings
        graph_query = GraphRAGQuery(graph=neo4j_graph)
        
        embeddings = EntityEmbeddings(
            graph=neo4j_graph,
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )
        
        # Khởi tạo Hybrid Retriever
        hybrid_retriever = HybridRetriever(
            graph_query=graph_query,
            embeddings=embeddings
        )
        
        context_builder = ContextBuilder()
        
        graph_rag = GeminiRAG(
            hybrid_retriever=hybrid_retriever,
            context_builder=context_builder,
            api_key=api_key,
            model_name="gemini-2.5-flash"
        )
        
        print("Graph RAG initialized successfully")
        
    except Exception as e:
        print(f"Không thể khởi tạo Graph RAG: {e}")
        print("Router sẽ chỉ sử dụng Traditional RAG")
        graph_rag = None
    
    # Tạo router
    print("\nTạo router...")
    router = IntelligentRAGRouter(
        traditional_rag=traditional_rag,
        graph_rag=graph_rag,
        api_key=api_key,
        classifier_model="gemini-2.5-flash"
    )
    
    print("\nRouter đã sẵn sàng!")
    return router


if __name__ == "__main__":
    # Ví dụ sử dụng
    router = create_router_with_defaults()
    
    # Test queries
    test_questions = [
        "Mức phạt không đội mũ bảo hiểm là bao nhiêu?",  # Should use Traditional RAG
        "So sánh mức phạt vượt đèn đỏ giữa xe máy và ô tô",  # Should use Graph RAG
        "Tại sao phải giữ khoảng cách an toàn khi tham gia giao thông?"  # Should use Graph RAG
    ]
    for question in test_questions:
        print("\n" + "="*70)
        result = router.route_query(question)
        print(f"\nCâu hỏi: {question}")
        print(f"Hệ thống: {result['system_used']}")
        print(f"Loại: {result['query_type']}")
        print(f"\nCâu trả lời:\n{result['answer'][:200]}...")
    
    # Hoặc chạy interactive mode
    # router.interactive_query()
