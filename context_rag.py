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
        use_rerank: bool = False,
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
        q_emb = self.embed_model.encode([query], convert_to_numpy=True).tolist()
        content_filter = None

        if "đi bộ" in query.lower():
            content_filter = {"$contains": "đi bộ"}
        if "xe gắn máy" in query.lower():
            content_filter = {"$contains": "xe gắn máy"}
        res = self.collection.query(
            query_embeddings=q_emb,
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where_document=content_filter
        )

        hits = []
        if res["ids"] and res["ids"][0]:
            for i in range(len(res["ids"][0])):
                hits.append(
                    {
                        "id": res["ids"][0][i],
                        "content": res["documents"][0][i],
                        "metadata": res["metadatas"][0][i],
                        "score_dense": 1.0 - res["distances"][0][i],
                    }
                )

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
        use_rerank: bool = True,
        top_n_final: int = 10,
    ):
        # Ánh xạ thay thế trực tiếp 1–1
        replace_map = {
            "rượu": "nồng độ cồn",
            "bia": "nồng độ cồn",
            "xe máy": "xe gắn máy",
            "xử lý": "phạt",
            "có sao không": "bị phạt không",
            "băng qua đường": "đi bộ qua đường",
            "đi qua đường": "đi bộ qua đường",
            "xe con": "xe ô tô",
            "lái tàu hỏa": "nhân viên đường sắt",
            "gas": "hàng hoá dễ cháy",
            "bỏ chạy": "bỏ trốn",
            "chiếu lên": "gây ảnh hưởng",
        }

        for src, dst in replace_map.items():
            if src in query:
                query = query.replace(src, dst)

        # Các rule phụ thuộc vào ngữ cảnh "phạt"
        if "thế nào" in query and "phạt" not in query:
            query = query.replace("thế nào", "phạt thế nào")
        if "vi phạm" in query and "phạt" not in query:
            query = query.replace("vi phạm", "phạt")
        xe_uu_tien_map = [
            "xe cứu thương",
            "xe cứu hỏa",
            "xe chữa cháy",
            "xe công an",
            "xe cảnh sát",
            "xe quân sự",
            "xe hộ tống",
            "xe dẫn đoàn",
            "xe ưu tiên",
        ]
        for kw in xe_uu_tien_map:
            if kw in query:
                query = query.replace(kw, "xe ưu tiên")
        # Nhóm xe máy chuyên dùng
        specialized_vehicles = [
            "máy xúc",
            "xe đào",
            "xe ủi",
            "xe lu",
            "xe san",
            "xe cẩu",
            "xe nâng",
            "xe gặt đập liên hợp",
            "máy kéo",
            "máy cày",
        ]
        for kw in specialized_vehicles:
            if kw in query:
                query = query.replace(kw, "xe máy chuyên dùng")

        # Nhóm từ đồng nghĩa/biến thể của xe gắn máy
        xe_gan_may_map = [
            "xe tay ga",
            "xe ga",
            "xe scooter",
            "xe số",
            "xe máy điện",
            "xe tay ga điện",
            "xe cub",
            "xe dream",
            "xe wave",
            "xe sirius",
            "xe vision",
            "xe lead",
            "xe air blade",
            "xe máy",  # để cuối cùng để không phá rule ở trên
        ]
        for kw in xe_gan_may_map:
            if kw in query:
                query = query.replace(kw, "xe gắn máy")

        # 1) dense retrieval
        dense_hits = self.dense_search(query, top_k=top_k_dense)

        if "xe gắn máy" in query and "xe máy chuyên dùng" not in query:
            dense_hits = [
                h for h in dense_hits if "xe máy chuyên dùng" not in h["content"]
            ]

        if (
            "ô tô" in query
            and "ô tô kinh doanh" not in query
            and "ô tô chở trẻ" not in query
        ):
            dense_hits = [
                h
                for h in dense_hits
                if "ô tô kinh doanh" not in h["content"]
                and "ô tô chở trẻ" not in h["content"]
            ]

        if "ô tô" in query and "đường" in query:
            dense_hits = [
                h
                for h in dense_hits
                if "lĩnh vực hàng hải" not in h["metadata"]["law_name"]
            ]

        # 2) rerank nếu cần
        if use_rerank:
            selected_hits = self.rerank_vi(query, dense_hits, top_n=top_n_final)
        else:
            selected_hits = dense_hits[:top_n_final]

        # 3) build context
        context = self.build_context_from_hits(selected_hits)

        return {
            "context": context,
            "hits": selected_hits,
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
        
        # Debug: in độ dài prompt
        prompt_length = len(prompt)
        print(f"\n[DEBUG] Prompt length: {prompt_length} chars")
        print(f"[DEBUG] Max output tokens: {max_tokens}")
        
        resp = self.gemini_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.5,
                top_p=0.9,
            ),
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
        
        # Debug: in thông tin về response
        print(f"[DEBUG] Response candidates: {len(resp.candidates)}")
        if resp.candidates:
            candidate = resp.candidates[0]
            print(f"[DEBUG] Finish reason: {candidate.finish_reason}")
            print(f"[DEBUG] Safety ratings: {candidate.safety_ratings}")
            if hasattr(candidate, 'token_count'):
                print(f"[DEBUG] Token count: {candidate.token_count}")
        
        response_text = (resp.text or "").strip()
        print(f"[DEBUG] Response length: {len(response_text)} chars")
        
        return response_text
    def build_llm_context_from_hits(self,hits):
        """
        hits: danh sách kết quả retrieve từ rag_retrieve(...)
        Mỗi phần tử có dạng:
        {
            "doc": {
                "content": "...",
                "metadata": {
                    "law_name": "...",
                    "law_code": "...",
                    "article": "...",
                    "clauses": [...],
                    ...
                }
            },
            ...
        }
        """
        blocks = []
        for idx, h in enumerate(hits, start=1):
            doc = h["doc"] if isinstance(h, dict) and "doc" in h else h
            content = doc["content"]
            md = doc["metadata"]

            law_name = md.get("law_name", "Không rõ tên luật")
            law_code = md.get("law_code", "Không rõ mã luật")
            article = md.get("article", "Không rõ điều")
            clauses = md.get("clauses", [])
            clause = clauses[0] if clauses else "Không rõ khoản"

            block = f"""[{idx}]
                        Luật: {law_name} ({law_code})
                        Điều: {article}
                        Khoản: {clause}
                        Nội dung:
                        {content}
                    """
            blocks.append(block)

        return "\n\n".join(blocks)
    
    def llm_answer_from_rag(self, question: str, rag_result: dict) -> str:
        hits = rag_result["hits"]
        context_text = self.build_llm_context_from_hits(hits)

        parts = []

        # PROMPT DUY NHẤT ĐÚNG NHƯ BẠN YÊU CẦU
        parts.append("""Bạn là một chuyên gia luật giao thông Việt Nam. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp.

    HƯỚNG DẪN:
    - Chỉ sử dụng thông tin từ context được cung cấp
    - Trả lời chính xác, có căn cứ, đi thẳng vào vấn đề 
    - Nếu không có đủ thông tin, hãy nói rõ
    - Sử dụng tiếng Việt tự nhiên và dễ hiểu
    YÊU CẦU TRẢ LỜI:
    - Trả lời theo dạng đoạn văn mạch lạc, không trả lời với câu mở đầu "dựa vào ngữ cảnh..." hãy trả lời trực tiếp
    - Nêu rõ:
    + Quy định / nội dung chính
    + Dẫn chứng từ context (nếu có)
    + Giải thích
    - Phải có kết luận cuối cùng (kết luận highlight lên để người đọc dễ nhận biết)
    - Câu trả lời phải logic.
    - Cuối câu ghi nguồn: Ví dụ: Nguồn: Nghị định 168/2024/NĐ-CP, Điều 5, Khoản 2, Luật Trật tự, an toàn giao thông đường bộ
    - Nếu context không đủ để xác định rõ vi phạm và căn cứ pháp lý, bắt buộc phải dùng MỘT trong các cụm sau trong câu trả lời:
    "không đủ thông tin"
    "không đủ dữ kiện".""")

        # GHÉP CONTEXT + CÂU HỎI
        parts.append(f"CÂU HỎI:\n\"\"\"{question}\"\"\"")
        parts.append(f"CONTEXT (CÁC ĐOẠN LUẬT):\n{context_text}")

        prompt = "\n\n".join(parts)

        answer = self.call_gemini(prompt, max_tokens=2048)
        return answer
    def is_insufficient_answer(self, answer: str) -> bool:
        text = answer.lower()
        keywords = [
            "không đủ thông tin",
            "không đủ dữ kiện",
        ]
        return any(kw in text for kw in keywords)
    def rewrite_question_for_rag(self, original_question: str) -> str:
        prompt = f"""
    Bạn là một chuyên gia pháp lý chuyên sâu về Luật Giao thông của Việt Nam và các Nghị định xử phạt vi phạm hành chính trong lĩnh vực giao thông.

    Nhiệm vụ:
    - Hãy viết lại câu hỏi dân dã của người dùng thành một câu hỏi pháp lý chuẩn mực và phạm vi rõ ràng (Đường bộ, đường thuỷ, hàng hải, hàng không, đường sắt), có loại xe rõ ràng.
    - Không được thay đổi ý nghĩa chính của câu hỏi.
    - Trả về duy nhất 1 câu hỏi đã viết lại, không thêm giải thích.

    Câu hỏi gốc của người dùng:
    \"\"\"{original_question}\"\"\"

    Câu hỏi đã viết lại:
    """
        new_q = self.call_gemini(prompt, max_tokens=2048)
        return new_q.strip()
    xe_uu_tien_map = [
        "xe cứu thương",
        "xe cứu hỏa",
        "xe chữa cháy",
        "xe công an",
        "xe cảnh sát",
        "xe quân sự",
        "xe hộ tống",
        "xe dẫn đoàn",
        "xe ưu tiên",
    ]
    def is_specialized_question(self, question: str) -> bool:
        q_lower = question.lower()
        return any(kw.lower() in q_lower for kw in self.xe_uu_tien_map)
    def rag_qa(self, question: str,
            top_k_dense: int = 30,
            use_rerank: bool = False,
            top_n_final: int = 5,
            enable_rewrite: bool = True):

        if self.is_specialized_question(question):
            top_n_final = 10
        rag_result = self.rag_retrieve(
            query=question,
            top_k_dense=top_k_dense,
            use_rerank=use_rerank,
            top_n_final=top_n_final,
        )

        answer = self.llm_answer_from_rag(question, rag_result)

        # Nếu câu trả lời ổn (không "không đủ thông tin"), trả về format thống nhất
        if not (enable_rewrite and self.is_insufficient_answer(answer)):
            return {
                "question": question,
                "answer": answer,
                "context": rag_result["context"],
                "hits": rag_result["hits"],
                "normalized_query": question,  # Query không bị rewrite
            }
        print("sửa lại câu hỏi")
        # ===== LẦN 2: REWRITE QUESTION CHO RAG =====
        rewritten_q = self.rewrite_question_for_rag(question)
        print(rewritten_q)
        rag_result_2 = self.rag_retrieve(
            query=rewritten_q,
            top_k_dense=top_k_dense,
            use_rerank=use_rerank,
            top_n_final=top_n_final,
        )

        # Lưu lại cho debug nếu muốn (ví dụ thêm vào dict)
        rag_result_2["rewritten_query"] = rewritten_q

        # GỌI LẠI LLM: vẫn trả lời theo CÂU HỎI GỐC
        answer_2 = self.llm_answer_from_rag(question, rag_result_2)

        return {
            "question": question,
            "answer": answer_2,
            "context": rag_result_2["context"],
            "hits": rag_result_2["hits"],
            "normalized_query": rewritten_q,  # Query đã được rewrite
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
        default="Đang lái ô tô đi làm, tôi nghe tiếng còi xe cứu thương phía sau nhưng đường đông quá tôi cứ đi bình thường không nhường. Như vậy phạt bao nhiêu?",
        help="Câu hỏi test"
    )
    
    args = parser.parse_args()
    
    # Get API key from args or env
    api_key = args.gemini_api_key or os.environ.get("GOOGLE_API_KEY")

    # Init RAG system
    rag_system = ContextRAG(
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        gemini_api_key=api_key,
        gemini_model=args.gemini_model,
        use_rerank=False,
    )
    
    # Test question
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    print(f"{'='*60}\n")
    
    result = rag_system.rag_qa(args.question)

    print("ANSWER:")
    print(result["answer"])
    print(f"\n{'='*60}")
    print(f"Retrieved {len(result['hits'])} documents")
    print(f"Normalized query: {result['normalized_query']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
