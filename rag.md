# RAG - Retrieval-Augmented Generation

## Tổng quan

**RAG (Retrieval-Augmented Generation)** là kỹ thuật kết hợp retrieval (tìm kiếm thông tin) với generation (sinh văn bản) để cải thiện khả năng trả lời câu hỏi của Large Language Models (LLMs).

### Vấn đề RAG giải quyết

1. **Hallucination** - LLM tạo ra thông tin sai lệch
2. **Knowledge cutoff** - LLM không biết thông tin mới sau thời điểm training
3. **Domain-specific knowledge** - LLM thiếu kiến thức chuyên sâu về lĩnh vực cụ thể
4. **Citation** - Khó trích dẫn nguồn cho câu trả lời

### Ý tưởng cốt lõi

Thay vì LLM trả lời dựa hoàn toàn vào knowledge được học trong quá trình training, RAG:
1. **Retrieve** (Tìm kiếm) thông tin liên quan từ knowledge base
2. **Augment** (Bổ sung) prompt với thông tin này
3. **Generate** (Sinh) câu trả lời dựa trên context được cung cấp

---

## Kiến trúc RAG

```
┌─────────────┐
│   Query     │  "Cách mạng tháng Tám thành công do đâu?"
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│   1. RETRIEVAL                      │
│   - Vector Search / Graph Search    │
│   - Find relevant documents         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   2. CONTEXT BUILDING               │
│   - Format retrieved docs           │
│   - Build structured prompt         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   3. GENERATION                     │
│   - LLM (Gemini/GPT/etc)           │
│   - Generate answer from context    │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Answer    │  "Cách mạng tháng Tám thành công do..."
└─────────────┘
```

---

## Traditional RAG

### Workflow

```
User Query
    │
    ▼
┌─────────────────────┐
│ Embedding Model     │  Encode query → vector
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Vector Database     │  Search similar chunks
│ (ChromaDB/FAISS)    │  (cosine similarity)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Top-K Chunks        │  Retrieved documents
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Context Builder     │  Format into prompt
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ LLM (Gemini)        │  Generate answer
└─────────┬───────────┘
          │
          ▼
     Final Answer
```

### Chi tiết các bước

#### 1. Document Chunking
```
Original Document
    ↓
Split into chunks (300-500 words)
    ↓
Add metadata (page, chapter, etc.)
```

**Example:**
```json
{
  "id": "chunk_0009",
  "content": "Cách mạng tháng Tám 1945 đã thành công...",
  "metadata": {
    "page_start": 14,
    "page_end": 22,
    "chapter": "Chương I",
    "word_count": 450
  }
}
```

#### 2. Embedding Generation
```
Text → Embedding Model → Dense Vector [768 dims]
```

**Model:** `paraphrase-multilingual-mpnet-base-v2`
- Input: Vietnamese text
- Output: 768-dimensional vector
- Normalized to unit length

**Example:**
```python
text = "Cách mạng tháng Tám 1945"
embedding = model.encode(text)
# embedding: [-0.0548, 0.0858, -0.0051, ..., 0.0421]
# shape: (768,)
# norm: 1.0
```

#### 3. Vector Indexing
```
All chunks → Batch embedding → Store in ChromaDB
```

**ChromaDB Collection:**
- Name: `vietnamese_history_chunks`
- Total chunks: 288
- Metric: Cosine similarity
- Storage: Persistent (./chroma_db)

#### 4. Query Processing
```
User Query
    ↓
Encode to vector
    ↓
Search in ChromaDB (cosine similarity)
    ↓
Top-K most similar chunks
```

**Example:**
```python
query = "Cách mạng tháng Tám thành công do đâu?"
query_vector = model.encode(query)

results = chromadb.query(
    query_embeddings=[query_vector],
    n_results=5
)

# Results sorted by similarity:
# 1. chunk_0009 - similarity: 0.643
# 2. chunk_0136 - similarity: 0.630
# 3. chunk_0142 - similarity: 0.625
```

#### 5. Context Building
```
Retrieved Chunks
    ↓
Format with metadata
    ↓
Build structured prompt
```

**Prompt Structure:**
```
HƯỚNG DẪN:
- Chỉ sử dụng thông tin từ các đoạn văn được cung cấp
- Trả lời chính xác, chi tiết
- Trích dẫn số thứ tự đoạn văn

CÁC ĐOẠN VĂN LIÊN QUAN:

--- Đoạn 1 ---
Nguồn: Trang 14-22
Độ liên quan: 0.643

[content of chunk 1]

--- Đoạn 2 ---
...

CÂU HỎI: [user question]

TRẢ LỜI:
```

#### 6. LLM Generation
```
Prompt → Gemini API → Generated Answer
```

**Configuration:**
```python
model = "gemini-2.5-flash"
temperature = 0.7
max_tokens = 8192
```

---

## Graph RAG

### Khác biệt với Traditional RAG

| Aspect | Traditional RAG | Graph RAG |
|--------|----------------|-----------|
| **Data Structure** | Flat chunks | Knowledge Graph (entities + relationships) |
| **Retrieval** | Vector similarity | Hybrid (vector + graph traversal) |
| **Context** | Independent chunks | Connected entities with relationships |
| **Strengths** | Simple, fast | Rich context, relationships |
| **Complexity** | Low | High |

### Workflow

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│ 1. SEMANTIC SEARCH              │
│    Query → Find seed entities    │
│    (Vector similarity in Neo4j) │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 2. GRAPH EXPANSION              │
│    Traverse relationships        │
│    Get connected entities        │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 3. HYBRID SCORING               │
│    - Vector similarity: 35%     │
│    - Graph proximity: 25%       │
│    - Entity type match: 25%     │
│    - Seed quality: 15%          │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 4. CONTEXT WITH RELATIONSHIPS   │
│    - Top entities               │
│    - Relationships between them │
│    - Source page content        │
└─────────┬───────────────────────┘
          │
          ▼
┌─────────────────────────────────┐
│ 5. LLM GENERATION               │
└─────────┬───────────────────────┘
          │
          ▼
     Final Answer
```

### Knowledge Graph Structure

#### Entities (Nodes)
```
PERSON: Hồ Chí Minh, Võ Nguyên Giáp, ...
LOCATION: Hà Nội, Điện Biên Phủ, ...
ORGANIZATION: Việt Minh, Đảng Cộng sản, ...
EVENT: Cách mạng tháng Tám, ...
TIME: 1945, tháng 8 năm 1945, ...
```

**Properties:**
- `name`: Tên entity
- `description`: Mô tả
- `embedding`: Vector 768 chiều
- `first_seen_page`: Trang xuất hiện đầu tiên
- `first_seen_chapter`: Chương

#### Relationships (Edges)
```
(Hồ Chí Minh) -[LÃNH_ĐẠO]-> (Cách mạng tháng Tám)
(Cách mạng tháng Tám) -[DIỄN_RA_TẠI]-> (Hà Nội)
(Việt Minh) -[THAM_GIA]-> (Cách mạng tháng Tám)
```

**Properties:**
- `type`: Loại quan hệ
- `description`: Mô tả chi tiết

### Example: Graph RAG Query

**Query:** "Ai lãnh đạo Cách mạng tháng Tám?"

**Step 1: Semantic Search**
```cypher
// Find entities similar to query
MATCH (e)
WHERE e.embedding IS NOT NULL
WITH e, gds.similarity.cosine(e.embedding, $query_embedding) AS similarity
ORDER BY similarity DESC
LIMIT 5

Results:
1. Hồ Chí Minh (PERSON) - 0.72
2. Cách mạng tháng Tám (EVENT) - 0.68
3. Đảng Cộng sản (ORGANIZATION) - 0.65
```

**Step 2: Graph Expansion**
```cypher
// Get relationships
MATCH (seed)-[r]-(related)
WHERE seed.name IN ['Hồ Chí Minh', 'Cách mạng tháng Tám']
RETURN seed, r, related

Results:
(Hồ Chí Minh) -[LÃNH_ĐẠO]-> (Cách mạng tháng Tám)
(Hồ Chí Minh) -[CHỦ_TỊCH]-> (Việt Nam Dân chủ Cộng hòa)
(Cách mạng tháng Tám) -[THÀNH_CÔNG]-> (2/9/1945)
```

**Step 3: Context**
```
ENTITIES:
1. Hồ Chí Minh (PERSON) - score: 0.85
   - Lãnh tụ của Đảng Cộng sản và nhân dân Việt Nam
   
2. Cách mạng tháng Tám (EVENT) - score: 0.78
   - Cuộc cách mạng giành độc lập năm 1945

RELATIONSHIPS:
1. Hồ Chí Minh → [LÃNH_ĐẠO] → Cách mạng tháng Tám
   Chi tiết: Chủ tịch Hồ Chí Minh đã lãnh đạo...

SOURCE CONTENT:
1. Trang 14-22:
   [actual page content from chapter10.md]
```

---

## So sánh Traditional RAG vs Graph RAG

### Performance

| Metric | Traditional RAG | Graph RAG |
|--------|----------------|-----------|
| **Retrieval Time** | 0.22s | 7.07s |
| **Context Building** | Instant | 2-3s |
| **Generation Time** | 3.79s | 23s |
| **Total Time** | ~4s | ~30s |
| **Prompt Tokens** | 3,077 | ~5,000-7,000 |

### Context Quality

**Traditional RAG:**
- ✅ Nhanh, đơn giản
- ✅ Scaling tốt với large datasets
- ❌ Chunks độc lập, thiếu context
- ❌ Không hiểu relationships

**Graph RAG:**
- ✅ Rich context với relationships
- ✅ Hiểu connections giữa entities
- ✅ Trả lời phức tạp tốt hơn (multi-hop reasoning)
- ❌ Chậm hơn
- ❌ Phức tạp hơn để setup
- ❌ Requires graph construction

### Use Cases

**Traditional RAG phù hợp khi:**
- Cần tốc độ cao
- Documents độc lập
- Câu hỏi đơn giản, factual
- Large-scale production

**Graph RAG phù hợp khi:**
- Cần relationships
- Multi-hop reasoning
- Complex queries
- Domain với nhiều entities liên kết
- Quality > Speed

---

## Implementation Details - Project này

### Architecture Overview

```
┌────────────────────────────────────────────────────────────┐
│                    USER QUERY                              │
└────────────────────┬───────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────────┐    ┌──────────────────┐
│ TRADITIONAL RAG  │    │   GRAPH RAG      │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────┐
│   ChromaDB       │    │     Neo4j        │
│  288 chunks      │    │  Entities + Rels │
└────────┬─────────┘    └────────┬─────────┘
         │                       │
         └───────────┬───────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  Gemini API      │
            │ gemini-2.5-flash │
            └────────┬─────────┘
                     │
                     ▼
            ┌──────────────────┐
            │  FINAL ANSWER    │
            └──────────────────┘
```

### Files Structure

#### Traditional RAG
```
traditional_rag_embeddings.py    # Embedding model wrapper
traditional_rag_vectorstore.py   # ChromaDB integration
traditional_rag_retriever.py     # Retrieval logic
traditional_rag_context.py       # Context formatting
traditional_rag_gemini.py        # End-to-end pipeline
```

#### Graph RAG
```
graph_rag_embeddings.py          # Entity embeddings in Neo4j
graph_rag_query.py               # Cypher queries
graph_rag_hybrid.py              # Hybrid retrieval
graph_rag_context.py             # Context with relationships
graph_rag_gemini.py              # End-to-end pipeline
```

### Data Flow - Traditional RAG

```python
# 1. Initialize
from traditional_rag_gemini import TraditionalGeminiRAG

rag = TraditionalGeminiRAG()

# 2. Query
result = rag.answer(
    question="Cách mạng tháng Tám thành công do đâu?",
    top_k=5
)

# 3. Result
{
    'question': '...',
    'answer': '...',
    'chunks': [...],
    'retrieval_time': 0.22,
    'generation_time': 3.79,
    'total_time': 4.01,
    'usage': {
        'prompt_tokens': 3077,
        'completion_tokens': 81
    }
}
```

### Data Flow - Graph RAG

```python
# 1. Initialize
from graph_rag_gemini import GeminiRAG
from graph_rag_hybrid import HybridRetriever
from graph_rag_context import ContextBuilder

retriever = HybridRetriever(graph, embeddings, query_module)
context_builder = ContextBuilder()
rag = GeminiRAG(retriever, context_builder)

# 2. Query
result = rag.generate_answer(
    question="Cách mạng tháng Tám thành công do đâu?",
    retrieval_params={'top_k': 10}
)

# 3. Result
{
    'question': '...',
    'answer': '...',
    'context': {
        'entities': [...],
        'relationships': [...],
        'sources': [...]
    },
    'retrieval_time': 7.07,
    'generation_time': 23.0,
    'total_time': 29.95
}
```

---

## Configuration & Parameters

### Embeddings
```python
model_name = "paraphrase-multilingual-mpnet-base-v2"
dimension = 768
device = "cuda:0"  # or "cpu"
```

### ChromaDB (Traditional RAG)
```python
collection_name = "vietnamese_history_chunks"
persist_directory = "./chroma_db"
metric = "cosine"
total_chunks = 288
```

### Neo4j (Graph RAG)
```python
uri = "neo4j+s://xxx.databases.neo4j.io"
entities = 1000+
relationships = 2000+
types = ["PERSON", "LOCATION", "ORGANIZATION", "EVENT", "TIME"]
```

### Gemini API
```python
model = "gemini-2.5-flash"
temperature = 0.7
max_output_tokens = 8192
```

### Hybrid Scoring (Graph RAG)
```python
weights = {
    'vector_similarity': 0.35,
    'graph_proximity': 0.25,
    'type_match': 0.25,
    'seed_quality': 0.15
}
```

---

## Best Practices

### 1. Chunking Strategy
```
Chunk Size: 300-500 words
Overlap: 50-100 words
Metadata: page, chapter, word_count
```

### 2. Retrieval Parameters
```
Traditional RAG:
- top_k: 5-10 chunks
- min_similarity: 0.5-0.6

Graph RAG:
- top_k: 10-15 entities
- expansion_depth: 1-2 hops
```

### 3. Prompt Engineering
```
✅ Clear instructions
✅ Structured format
✅ Examples khi cần
✅ Citation requirements
```

### 4. Error Handling
```python
try:
    result = rag.answer(question)
except Exception as e:
    # Fallback logic
    return {"answer": "Không thể trả lời...", "error": str(e)}
```

---

## Performance Optimization

### Traditional RAG
1. **Batch embedding** cho indexing
2. **Cache embeddings** trong memory
3. **Limit prompt length** để tránh token overflow
4. **Async generation** cho multiple queries

### Graph RAG
1. **Index properties** trong Neo4j
2. **Limit expansion depth** để avoid exponential growth
3. **Cache frequent queries**
4. **Parallel graph queries** khi có thể

---

## Evaluation Metrics

### Retrieval Quality
```python
# Precision@K
relevant_in_topk / k

# Recall@K
relevant_in_topk / total_relevant

# MRR (Mean Reciprocal Rank)
1 / rank_of_first_relevant
```

### Generation Quality
```python
# BLEU score
compare_with_reference_answer()

# Human evaluation
- Accuracy
- Completeness
- Citation quality
```

### System Performance
```python
# Latency
- Retrieval time
- Generation time
- Total time

# Token efficiency
- Prompt tokens
- Completion tokens
- Cost per query
```

---

## Troubleshooting

### Common Issues

**1. Empty retrieval results**
```
Problem: No chunks/entities retrieved
Solution: 
- Lower min_similarity threshold
- Increase top_k
- Check embedding model
```

**2. Irrelevant results**
```
Problem: Retrieved chunks not related to query
Solution:
- Improve chunking strategy
- Better metadata
- Query expansion
```

**3. Token limit exceeded**
```
Problem: Prompt too long
Solution:
- Reduce top_k
- Truncate chunks
- Summarize context
```

**4. Slow performance**
```
Problem: High latency
Solution:
- Cache embeddings
- Index optimization
- Reduce expansion depth (Graph RAG)
```

---

## Future Improvements

### Traditional RAG
- [ ] Hybrid search (keyword + semantic)
- [ ] Reranking models
- [ ] Query expansion
- [ ] Multi-modal embeddings

### Graph RAG
- [ ] Dynamic graph construction
- [ ] Temporal reasoning
- [ ] Multi-hop path finding
- [ ] Graph neural networks

### Both
- [ ] Fine-tuned embedding models
- [ ] Custom prompt templates
- [ ] A/B testing framework
- [ ] User feedback loop

---

## References

### Papers
- RAG: "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- Graph RAG: "Graph-RAG: Knowledge Graph Enhanced RAG" (Microsoft Research, 2024)

### Tools
- ChromaDB: https://www.trychroma.com/
- Neo4j: https://neo4j.com/
- Sentence Transformers: https://www.sbert.net/
- Google Gemini: https://ai.google.dev/

### Our Implementation
- Source: `/home/duo/work/DL/btl/`
- Data: Vietnamese History (Chapter 10: 1945-1950)
- Language: Vietnamese
- Models: paraphrase-multilingual-mpnet-base-v2 + gemini-2.5-flash

---

## Conclusion

**RAG transforms LLMs** từ knowledge stored trong parameters sang knowledge retrieved from external sources. Điều này cho phép:
- ✅ Accurate, grounded answers
- ✅ Up-to-date information
- ✅ Domain expertise
- ✅ Traceable citations

**Chọn approach phù hợp:**
- **Traditional RAG** cho production systems cần speed
- **Graph RAG** cho complex domains cần relationships

**Key takeaway:** RAG không phải là silver bullet, nhưng là công cụ quan trọng để build reliable, useful AI applications.
