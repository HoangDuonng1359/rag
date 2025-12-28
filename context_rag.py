"""
Script để retrieve context từ ChromaDB và tạo câu trả lời với Gemini
"""

import os
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)

class ContextRAG:
    def __init__(
        self,
        chroma_dir: str = "chroma_db",
        collection_name: str = "vn_traffic_law",
        embedding_model_name: str = "truro7/vn-law-embedding",
        rerank_model_name: str = "huynhdat543/VietNamese_law_rerank",
        gemini_api_key: str = None,
        gemini_model: str = "gemini-2.5-flash",
        use_rerank: bool = True,
        embedding_model=None,  # Cho phép pass model từ bên ngoài
    ):
        """
        Khởi tạo Context RAG system.
        
        Args:
            chroma_dir: Thư mục chứa ChromaDB
            collection_name: Tên collection trong ChromaDB
            embedding_model_name: Model embedding để encode query
            rerank_model_name: Model rerank để sắp xếp lại kết quả
            gemini_api_key: API key cho Gemini
            gemini_model: Tên model Gemini
            use_rerank: Có sử dụng rerank hay không
            embedding_model: Model đã được load từ bên ngoài (optional)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load hoặc sử dụng embedding model có sẵn
        if embedding_model is not None:
            print(f"Using shared embedding model (already loaded)")
            self.embed_model = embedding_model
        else:
            print(f"Loading embedding model: {embedding_model_name}")
            self.embed_model = SentenceTransformer(embedding_model_name, device=self.device)
        
        # Load ChromaDB
        print(f"Loading ChromaDB from: {chroma_dir}")
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_collection(collection_name)
        print(f"Collection size: {self.collection.count()}")
        
        # Load rerank model if needed
        self.use_rerank = use_rerank
        if use_rerank:
            print(f"Loading rerank model: {rerank_model_name}")
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(
                rerank_model_name,
                trust_remote_code=True,
            )
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                rerank_model_name,
                trust_remote_code=True,
            ).to(self.device)
            self.rerank_model.eval()
        
        # Configure Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel(gemini_model)
            print(f"Gemini configured with model: {gemini_model}")
        else:
            self.gemini_model = None
            print("Warning: Gemini API key not provided. Q&A feature disabled.")
    
    def dense_search(self, query: str, top_k: int = 30) -> List[Dict[str, Any]]:
        """
        Tìm kiếm dense vector trong ChromaDB.
        
        Args:
            query: Câu query
            top_k: Số lượng kết quả trả về
            
        Returns:
            List các hits với id, content, metadata, score
        """
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).tolist()
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        
        hits = []
        for i in range(len(res["ids"][0])):
            hits.append({
                "id": res["ids"][0][i],
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score_dense": 1.0 - res["distances"][0][i],  # cosine -> similarity
            })
        return hits
    
    @torch.no_grad()
    def rerank_vi(self, query: str, hits: List[Dict], top_n: int = 8) -> List[Dict]:
        """
        Rerank kết quả bằng Vietnamese law rerank model.
        
        Args:
            query: Câu query
            hits: List các hits từ dense search
            top_n: Số lượng kết quả cuối cùng
            
        Returns:
            List các hits đã được rerank và sắp xếp
        """
        if not hits:
            return []
        
        # Ghép query + passage thành 1 chuỗi
        texts = [f"[Q] {query}\n[P] {h['content']}" for h in hits]
        
        enc = self.rerank_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        
        outputs = self.rerank_model(**enc)
        logits = outputs.logits
        
        # Model này trả về regression score [batch, 1], không phải binary classification
        scores = logits.squeeze(-1)  # [batch, 1] -> [batch]
        scores = scores.detach().cpu().tolist()
        
        reranked = []
        for h, s in zip(hits, scores):
            h2 = h.copy()
            h2["score_rerank"] = float(s)
            reranked.append(h2)
        
        reranked_sorted = sorted(reranked, key=lambda x: x["score_rerank"], reverse=True)
        return reranked_sorted[:top_n]
    
    def build_context_from_hits(self, hits: List[Dict]) -> str:
        """
        Xây dựng context string từ list hits để đưa vào prompt.
        
        Args:
            hits: List các hits đã được filter/rerank
            
        Returns:
            String context với format rõ ràng
        """
        pieces = []
        
        for h in hits:
            meta = h.get("metadata") or {}
            article = meta.get("article") or meta.get("dieu")
            clause = meta.get("clause") or meta.get("khoan")
            source = meta.get("law_name") or meta.get("source") or meta.get("file_name")
            
            header_parts = []
            if source:
                header_parts.append(f"Văn bản: {source}")
            if article:
                header_parts.append(f"Điều: {article}")
            if clause:
                header_parts.append(f"Khoản: {clause}")
            
            header = " | ".join(header_parts) if header_parts else f"Đoạn trích (id={h['id']})"
            
            block = f"{header}\n{h['content'].strip()}\n"
            pieces.append(block)
        
        return "\n---\n".join(pieces)
    
    def rag_retrieve(
        self,
        query: str,
        top_k_dense: int = 30,
        top_n_final: int = 10,
    ) -> Dict[str, Any]:
        """
        Retrieve context từ ChromaDB với query.
        
        Args:
            query: Câu hỏi của user
            top_k_dense: Số lượng kết quả từ dense search
            top_n_final: Số lượng kết quả cuối cùng sau rerank
            
        Returns:
            Dict chứa context string và hits
        """
        # Chuẩn hóa query: "xe máy" -> "xe gắn máy"
        normalized_query = query
        if "xe máy" in query and "xe máy chuyên dùng" not in query:
            normalized_query = query.replace("xe máy", "xe gắn máy")
        
        # 1) Dense retrieval
        dense_hits = self.dense_search(normalized_query, top_k=top_k_dense)
        
        # Filter out "xe máy chuyên dùng" nếu query về xe gắn máy
        if "xe gắn máy" in normalized_query and "xe máy chuyên dùng" not in normalized_query:
            dense_hits = [
                h for h in dense_hits 
                if "xe máy chuyên dùng" not in h["content"]
            ]
        
        # 2) Rerank nếu cần
        if self.use_rerank:
            selected_hits = self.rerank_vi(normalized_query, dense_hits, top_n=top_n_final)
        else:
            selected_hits = dense_hits[:top_n_final]
        
        # 3) Build context
        context = self.build_context_from_hits(selected_hits)
        
        return {
            "context": context,
            "hits": selected_hits,
            "normalized_query": normalized_query,
        }
    
    def call_gemini(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        Gọi Gemini API để tạo câu trả lời.
        
        Args:
            prompt: Prompt đầy đủ với context và question
            max_tokens: Số token tối đa cho response
            
        Returns:
            String câu trả lời từ Gemini
        """
        if not self.gemini_model:
            raise ValueError("Gemini model not configured. Please provide API key.")
        
        resp = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.4,
                top_p=0.9,
            ),
        )
        return (resp.text or "").strip()
    
    def build_qa_prompt(self, question: str, context: str) -> str:
        """
        Xây dựng prompt cho Q&A task với Gemini.
        
        Args:
            question: Câu hỏi của user
            context: Context từ RAG retrieve
            
        Returns:
            String prompt hoàn chỉnh
        """
        prompt = f"""Bạn là trợ lý pháp lý về giao thông đường bộ Việt Nam.

Chỉ dùng thông tin có trong CÁC ĐOẠN LUẬT (CONTEXT) để trả lời câu hỏi. Không được bịa thêm điều luật, số điều/khoản, mức phạt hoặc ví dụ ngoài context. Nếu không tìm thấy khoản nào phù hợp, trả lời "KHÔNG ĐỦ THÔNG TIN".

Lưu ý về thuật ngữ: "xe mô tô", "xe gắn máy", "xe máy" và "các loại xe tương tự" KHÁC với "xe máy chuyên dùng". Không được coi "xe máy chuyên dùng" là "xe mô tô/xe gắn máy/xe máy". Khi so sánh hành vi với context, phải căn cứ đúng loại phương tiện được nêu.

FORMAT TRẢ LỜI (bắt buộc, một đoạn duy nhất, phải chứa đủ cả KẾT LUẬN, CĂN CỨ và GIẢI THÍCH, nếu thiếu phần nào coi như trả lời sai):
- Bắt đầu bằng: "KẾT LUẬN: CÓ PHẠT." hoặc "KẾT LUẬN: KHÔNG PHẠT." hoặc "KẾT LUẬN: KHÔNG ĐỦ THÔNG TIN."
- Ngay sau đó: "Căn cứ: Luật [tên luật] ([mã luật]), Điều [số điều], Khoản [số khoản]."
- Sau đó, bắt đầu bằng "Giải thích:" và trong 1–3 câu:
  (1) Nêu ngắn gọn hành vi trong câu hỏi;
  (2) Nêu nội dung chính của khoản/điểm áp dụng trong context;
  (3) Nêu rõ vì sao hành vi đó (có hoặc không) thỏa điều kiện của khoản/điểm này (nếu kết luận CÓ PHẠT thì phải nói rõ TẠI SAO bị phạt).

CÂU HỎI GỐC:
\"\"\"{question}\"\"\"

CÁC ĐOẠN LUẬT (CONTEXT):
{context}
"""
        return prompt
    
    def answer_question(
        self,
        question: str,
        top_k_dense: int = 30,
        top_n_final: int = 10,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """
        Trả lời câu hỏi bằng RAG + Gemini.
        
        Args:
            question: Câu hỏi của user
            top_k_dense: Số lượng kết quả từ dense search
            top_n_final: Số lượng kết quả sau rerank
            max_tokens: Số token tối đa cho Gemini response
            
        Returns:
            Dict chứa answer, context, hits, và metadata
        """
        # 1) Retrieve context
        rag_result = self.rag_retrieve(
            query=question,
            top_k_dense=top_k_dense,
            top_n_final=top_n_final,
        )
        
        # 2) Build prompt
        prompt = self.build_qa_prompt(question, rag_result["context"])
        
        print("\n" + "="*60)
        print("PROMPT SENT TO GEMINI:")
        print("="*60)
        print(prompt)
        print("="*60 + "\n")
        
        # 3) Call Gemini
        answer = self.call_gemini(prompt, max_tokens=max_tokens)
        
        return {
            "question": question,
            "answer": answer,
            "context": rag_result["context"],
            "hits": rag_result["hits"],
            "normalized_query": rag_result["normalized_query"],
        }


def main():
    parser = argparse.ArgumentParser(description="RAG Context & Q&A với Gemini")
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default="chroma_db",
        help="Thư mục chứa ChromaDB"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="vn_traffic_law",
        help="Tên collection trong ChromaDB"
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=None,
        help="Gemini API key (hoặc set env GOOGLE_API_KEY)"
    )
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash",
        help="Tên model Gemini"
    )
    parser.add_argument(
        "--no_rerank",
        action="store_true",
        help="Tắt rerank"
    )
    parser.add_argument(
        "--question",
        type=str,
        default="Mức phạt vượt đèn đỏ đối với xe máy?",
        help="Câu hỏi test"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or env
    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")
    print(api_key)
    if not api_key:
        print("Warning: No Gemini API key provided. Set --gemini_api_key or GOOGLE_API_KEY env.")
    
    # Init RAG system
    rag_system = ContextRAG(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        gemini_api_key=api_key,
        gemini_model=args.gemini_model,
        use_rerank=not args.no_rerank,
    )
    
    # Test question
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")
    
    result = rag_system.answer_question(args.question)
    
    print("ANSWER:")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Retrieved {len(result['hits'])} documents")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
