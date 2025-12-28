"""
Script để embedding dữ liệu từ chunks_by_clause.jsonl và index vào ChromaDB
"""

import json
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import argparse

def clean_metadata(meta: dict) -> dict:
    """
    ChromaDB chỉ chấp nhận metadata có kiểu str, int, float, bool hoặc None.
    Chuyển list thành string để tránh lỗi.
    """
    clean = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, list):
            clean[k] = ",".join(map(str, v))
    return clean


def flush_batch(collection, model, ids, docs, metadatas, batch_size=8):
    """
    Embedding một batch documents và add vào ChromaDB collection.
    """
    if not ids:
        return
    
    print(f"Indexing batch size: {len(ids)}")
    embeddings = model.encode(
        docs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    ).tolist()

    collection.add(
        ids=ids,
        documents=docs,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def main(
    jsonl_path: str = "data/chunk/chunks_by_clause.jsonl",
    chroma_dir: str = "chroma_db",
    collection_name: str = "vn_traffic_law",
    batch_limit: int = 4000,
    embedding_batch_size: int = 8,
    model_name: str = "truro7/vn-law-embedding"
):
    """
    Main function để embedding và index dữ liệu.
    
    Args:
        jsonl_path: Đường dẫn đến file JSONL chứa chunks
        chroma_dir: Thư mục lưu ChromaDB
        collection_name: Tên collection trong ChromaDB
        batch_limit: Số lượng documents tối đa trong một batch để index
        embedding_batch_size: Batch size khi encode embedding
        model_name: Tên model embedding từ HuggingFace
    """
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load embedding model
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    # Init ChromaDB
    print(f"Initializing ChromaDB at: {chroma_dir}")
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Current collection size: {collection.count()}")
    
    # Read JSONL and batch index
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        raise FileNotFoundError(f"File not found: {jsonl_path}")
    
    print(f"Reading chunks from: {jsonl_path}")
    
    ids = []
    docs = []
    metadatas = []
    seen = {}  # Track duplicate IDs
    
    total_chunks = 0
    
    with open(jsonl_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            obj = json.loads(line)
            base_id = obj["id"]
            
            # Handle duplicate IDs
            if base_id in seen:
                seen[base_id] += 1
                uid = f"{base_id}_{seen[base_id]}"
            else:
                seen[base_id] = 0
                uid = base_id
            
            ids.append(uid)
            docs.append(obj["content"])
            raw_meta = obj.get("metadata", {})
            metadatas.append(clean_metadata(raw_meta))
            
            total_chunks += 1
            
            # Flush batch when reaching limit
            if len(ids) >= batch_limit:
                flush_batch(collection, model, ids, docs, metadatas, embedding_batch_size)
                ids, docs, metadatas = [], [], []
    
    # Flush remaining items
    if ids:
        flush_batch(collection, model, ids, docs, metadatas, embedding_batch_size)
    
    print(f"\n{'='*50}")
    print(f"Done! Total chunks indexed: {total_chunks}")
    print(f"Final collection size: {collection.count()}")
    print(f"ChromaDB saved at: {chroma_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding và index dữ liệu vào ChromaDB")
    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="data/chunk/chunks_by_clause.jsonl",
        help="Đường dẫn đến file JSONL"
    )
    parser.add_argument(
        "--chroma_dir",
        type=str,
        default="chroma_db",
        help="Thư mục lưu ChromaDB"
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="vn_traffic_law",
        help="Tên collection trong ChromaDB"
    )
    parser.add_argument(
        "--batch_limit",
        type=int,
        default=4000,
        help="Số lượng documents tối đa trong một batch"
    )
    parser.add_argument(
        "--embedding_batch_size",
        type=int,
        default=8,
        help="Batch size khi encode embedding (giảm nếu bị OOM)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="truro7/vn-law-embedding",
        help="Tên model embedding từ HuggingFace"
    )
    
    args = parser.parse_args()
    
    main(
        jsonl_path=args.jsonl_path,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection_name,
        batch_limit=args.batch_limit,
        embedding_batch_size=args.embedding_batch_size,
        model_name=args.model_name
    )
