# Graph RAG System - Hệ thống Truy vấn Lịch sử Việt Nam

## 📋 Tổng Quan

Đây là hệ thống **Graph RAG (Retrieval-Augmented Generation)** được xây dựng để trả lời các câu hỏi về lịch sử Việt Nam giai đoạn 1945-1975. Hệ thống kết hợp:

- **Knowledge Graph** (Neo4j) để lưu trữ các entities và relationships
- **Vector Embeddings** (Sentence Transformers) để tìm kiếm ngữ nghĩa
- **Hybrid Retrieval** kết hợp graph traversal và semantic search
- **LLM (Gemini)** để sinh câu trả lời tự nhiên
- **Source Context** từ file gốc chapter10.md

---

## 🏗️ Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────┐
│                    User Question                            │
│            "Cách mạng tháng Tám thành công do đâu?"        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   HybridRetriever          │
        │  (graph_rag_hybrid.py)     │
        └────────┬──────────┬────────┘
                 │          │
        ┌────────▼──────┐   │
        │ Vector Search │   │
        │  (Embeddings) │   │
        │               │   │
        │ EntityEmbedd- │   │
        │   ings        │   │
        └────────┬──────┘   │
                 │          │
                 │   ┌──────▼────────┐
                 │   │ Graph Travers │
                 │   │  (Cypher)     │
                 │   │               │
                 │   │ GraphRAGQuery │
                 │   └──────┬────────┘
                 │          │
                 └──────┬───┘
                        │
        ┌───────────────▼───────────────┐
        │    Hybrid Scoring & Ranking   │
        │  - Vector similarity: 35%     │
        │  - Graph proximity: 25%       │
        │  - Entity type: 25%           │
        │  - Seed quality: 15%          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │      ContextBuilder           │
        │   (graph_rag_context.py)      │
        │                               │
        │  1. Extract page numbers      │
        │  2. Load page content         │
        │  3. Format prompt             │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │        Gemini API             │
        │   (graph_rag_gemini.py)       │
        │                               │
        │  Generate natural answer      │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │         Answer                │
        │   "Cách mạng tháng Tám..."    │
        └───────────────────────────────┘
```

---

## 📁 Cấu Trúc Code

### 1. **graph_rag_embeddings.py** - Quản lý Vector Embeddings

**Chức năng:**
- Load model embedding (paraphrase-multilingual-mpnet-base-v2)
- Generate embeddings cho entities
- Semantic search dựa trên cosine similarity
- Lưu và load embeddings từ Neo4j

**Các method chính:**
```python
class EntityEmbeddings:
    def semantic_search(query, top_k=10)
        # Tìm entities tương tự với query
        # Returns: List[{name, type, description, similarity, first_seen_page}]
    
    def generate_embeddings_for_entities()
        # Generate embeddings cho tất cả entities trong graph
```

**Cập nhật quan trọng:**
- Query Neo4j bây giờ return `first_seen_page` và `first_seen_chapter`
- Kết quả semantic_search bao gồm page metadata

---

### 2. **graph_rag_query.py** - Graph Traversal với Cypher

**Chức năng:**
- Query Neo4j bằng Cypher
- Tìm related entities qua relationships
- Tìm paths giữa các entities

**Các method chính:**
```python
class GraphRAGQuery:
    def get_related_entities(entity_name, max_depth=1, limit=20)
        # Lấy các entities liên quan qua relationships
        # Returns: List[{name, type, description, relationships, distance}]
    
    def find_paths_between_entities(source, target, max_length=3)
        # Tìm đường đi giữa 2 entities
```

**Cập nhật quan trọng:**
- Tất cả queries bây giờ return `first_seen_page` và `first_seen_chapter`

---

### 3. **graph_rag_hybrid.py** - Hybrid Retrieval (CORE)

**Đây là thành phần quan trọng nhất**, kết hợp cả 2 phương pháp retrieval.

#### Luồng Hoạt Động Chi Tiết:

```python
def retrieve(question, top_k=15, vector_top_k=5, expansion_depth=1):
    """
    Phase 1: VECTOR SEARCH (Semantic Matching)
    ==========================================
    """
    # Bước 1.1: Infer question type
    q_type = infer_question_type(question)
    # Ví dụ: "Ai lãnh đạo..." -> WHO
    #        "Ở đâu..." -> WHERE
    #        "Khi nào..." -> WHEN
    
    # Bước 1.2: Semantic search
    seed_entities = embeddings.semantic_search(question, top_k=vector_top_k)
    # Returns: Top 5 entities có similarity cao nhất
    # Example: [
    #   {name: "Cách mạng Tháng Tám", similarity: 0.904, first_seen_page: 56},
    #   {name: "1945", similarity: 0.720, first_seen_page: 14},
    # ]
    
    """
    Phase 2: GRAPH EXPANSION (Context Enrichment)
    ==============================================
    """
    # Bước 2.1: Expand từ seed entities
    expanded = []
    for seed in seed_entities:
        # Lấy related entities qua graph relationships
        related = graph_query.get_related_entities(
            seed['name'], 
            max_depth=expansion_depth
        )
        expanded.extend(related)
    
    # Bước 2.2: Merge và deduplicate
    all_entities = seed_entities + expanded
    # unique by name
    
    """
    Phase 3: HYBRID SCORING (Intelligent Ranking)
    ==============================================
    """
    # Bước 3.1: Calculate hybrid score cho mỗi entity
    for entity in all_entities:
        # 3.1.1 Vector Score (35%)
        vector_score = seed_lookup.get(entity['name'], 0.0)
        
        # 3.1.2 Graph Score (25%)
        distance = entity.get('distance', 999)
        graph_score = max(0, 1.0 - distance * 0.2)
        
        # 3.1.3 Type Score (25%) - Question-aware
        type_weights = {
            'WHO': {'PERSON': 1.0, 'ORGANIZATION': 0.7},
            'WHERE': {'LOCATION': 1.0, 'EVENT': 0.5},
            'WHEN': {'TIME': 1.0, 'EVENT': 0.8},
            # ...
        }
        type_score = type_weights[q_type].get(entity['type'], 0.5)
        
        # 3.1.4 Source Score (15%) - Quality of seed
        source_similarity = entity.get('source_similarity', 0.5)
        
        # 3.1.5 Combine
        hybrid_score = (
            0.35 * vector_score +
            0.25 * graph_score +
            0.25 * type_score +
            0.15 * source_similarity
        )
        
        entity['score'] = hybrid_score
    
    # Bước 3.2: Sort by score và lấy top_k
    ranked = sorted(all_entities, key=lambda x: x['score'], reverse=True)
    top_entities = ranked[:top_k]
    
    """
    Phase 4: EXTRACT RELATIONSHIPS
    ================================
    """
    # Bước 4: Lấy relationships giữa các entities đã chọn
    relationships = []
    entity_names = {e['name'] for e in top_entities}
    
    for entity in top_entities:
        related = graph_query.get_related_entities(entity['name'])
        for rel in related:
            if rel['target'] in entity_names:
                relationships.append({
                    'source': entity['name'],
                    'target': rel['target'],
                    'type': rel['relationship_type'],
                    'description': rel.get('description', '')
                })
    
    return {
        'top_entities': top_entities,
        'relationships': relationships,
        'question_type': q_type
    }
```

**Cập nhật quan trọng:**
- Khi build entity dict, bây giờ copy `first_seen_page` và `first_seen_chapter` từ source
- Có 3 chỗ trong code cần update để preserve page metadata:
  1. Khi expand graph (line 213-220)
  2. Khi score expanded entities (line 277-285)
  3. Khi add seed entities (line 291-301)

---

### 4. **graph_rag_context.py** - Context Builder & Prompt Formatter

**Chức năng:**
- Build structured context từ retrieval results
- Load nội dung thực tế từ source file (chapter10.md)
- Format thành prompts cho LLM

#### Luồng Hoạt Động:

```python
def build_rag_context(question, retrieval_context):
    """
    Bước 1: Extract entities và relationships
    """
    entities = retrieval_context['top_entities'][:12]
    relationships = retrieval_context['relationships'][:20]
    
    """
    Bước 2: Extract và Load Page Content (QUAN TRỌNG!)
    """
    sources = _extract_sources_with_content(entities)
    
    # Chi tiết _extract_sources_with_content:
    sources_dict = {}
    for entity in entities:
        # Lấy page number
        page_num = entity.get('first_seen_page') or entity.get('page')
        
        if page_num:
            # Load nội dung thực tế từ file
            content = _load_page_content(page_num)
            
            sources_dict[page_num] = {
                'chapter': 10,
                'page': page_num,
                'citation': f"Chapter 10, Page {page_num}",
                'content': content  # Nội dung thật từ file!
            }
    
    return sorted(sources_dict.values(), key=lambda x: x['page'])
```

**Method `_load_page_content(page_number)`:**
```python
def _load_page_content(page_number, max_chars=1000):
    """Load nội dung từ data/chapter10.md"""
    
    # Check cache
    if page_number in self.page_cache:
        return self.page_cache[page_number]
    
    # Read file
    with open('data/chapter10.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract content cho page này
    # Pattern: --- Page X --- [content] --- Page X+1 ---
    pattern = rf"--- Page {page_number} ---\n(.*?)(?=--- Page {page_number + 1} ---|$)"
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        page_content = match.group(1).strip()
        # Truncate nếu quá dài
        if len(page_content) > max_chars:
            page_content = page_content[:max_chars] + "..."
        
        # Cache để tối ưu
        self.page_cache[page_number] = page_content
        return page_content
    
    return f"[Page {page_number} content not found]"
```

**Format Prompt cho Gemini:**
```python
def format_for_gemini(context, prompt_type="qa"):
    """
    Structure của prompt:
    
    1. SYSTEM INSTRUCTIONS (Vietnamese)
    2. ENTITIES với descriptions
    3. RELATIONSHIPS với directions
    4. CONNECTIONS (paths nếu có)
    5. SOURCE CONTENT (Nội dung thực tế từ file!) ← QUAN TRỌNG!
    6. QUESTION
    """
    
    parts = []
    
    # System instructions
    parts.append("""
    Bạn là chuyên gia lịch sử Việt Nam.
    - Chỉ sử dụng thông tin từ context
    - Trả lời chính xác, có căn cứ
    - Trích dẫn entities và relationships
    """)
    
    # Entities
    for entity in context['entities']:
        parts.append(f"{entity['name']} ({entity['type']})")
        parts.append(f"  Độ liên quan: {entity['score']:.3f}")
    
    # Relationships
    for rel in context['relationships']:
        parts.append(f"{rel['source']} --[{rel['type']}]--> {rel['target']}")
    
    # SOURCE CONTENT - Phần quan trọng nhất!
    if context.get('sources'):
        parts.append("\nSOURCE CONTENT (Nội dung nguồn):")
        for source in context['sources']:
            parts.append(f"\n{source['citation']}")
            parts.append(f"   {source['content'][:800]}...")
            # Đây là nội dung THỰC TẾ từ sách lịch sử!
    
    # Question
    parts.append(f"\nCÂU HỎI: {context['question']}")
    parts.append("\nTRẢ LỜI:")
    
    return "\n".join(parts)
```

---

### 5. **graph_rag_gemini.py** - LLM Integration

**Chức năng:**
- Kết nối với Gemini API
- Generate answer từ structured prompt
- Handle errors và safety settings

#### Luồng Hoạt Động:

```python
def generate_answer(question, prompt_type="qa", max_tokens=8192):
    """
    Step 1: RETRIEVE CONTEXT
    ========================
    """
    retrieval_context = retriever.retrieve(
        question=question,
        top_k=10,
        vector_top_k=5,
        expansion_depth=1
    )
    # Returns: {top_entities, relationships, question_type}
    
    """
    Step 2: BUILD RAG CONTEXT
    ==========================
    """
    rag_context = builder.build_rag_context(
        question=question,
        retrieval_context=retrieval_context,
        max_entities=10,
        max_relationships=15
    )
    # Returns: {entities, relationships, sources with content}
    
    """
    Step 3: FORMAT PROMPT
    =====================
    """
    prompt = builder.format_for_gemini(
        context=rag_context,
        prompt_type=prompt_type
    )
    # Returns: Complete prompt với instructions, context, sources
    
    """
    Step 4: GENERATE WITH GEMINI
    =============================
    """
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": max_tokens  # 8192 for full answer
        }
    )
    
    answer = response.text
    
    # Check if truncated
    if response.candidates[0].finish_reason != FinishReason.STOP:
        print("⚠️ Response may be incomplete")
    
    return {
        'question': question,
        'answer': answer,
        'context': rag_context,
        'metadata': {
            'entities_used': len(rag_context['entities']),
            'relationships_used': len(rag_context['relationships']),
            'prompt_tokens': response.usage_metadata.prompt_token_count,
            'completion_tokens': response.usage_metadata.candidates_token_count
        }
    }
```

---

## 🔄 Luồng Hoạt Động Tổng Thể (End-to-End)

### Example: "Cách mạng tháng Tám năm 1945 thành công do đâu?"

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Question Analysis                                  │
└─────────────────────────────────────────────────────────────┘
Input: "Cách mạng tháng Tám năm 1945 thành công do đâu?"

→ Infer question type: WHY/EXPLAIN
→ Keywords: ["Cách mạng tháng Tám", "1945", "thành công"]

┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Vector Search (Semantic)                           │
└─────────────────────────────────────────────────────────────┘
→ Generate query embedding
→ Search in entity embeddings
→ Top 5 results:
  1. "Cách mạng tháng Tám năm 1945" (EVENT) - similarity: 0.904
  2. "1945" (TIME) - similarity: 0.720
  3. "Kháng chiến chống Pháp 1945-1954" (EVENT) - 0.719
  4. "Cách mạng Tháng Tám" (EVENT) - 0.715
  5. "Việt Nam Dân chủ Cộng hòa" (ORG) - 0.680

┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Graph Expansion                                    │
└─────────────────────────────────────────────────────────────┘
For each seed entity:
  → Query: MATCH (seed)-[r*1..1]-(related)
  → Expand "Cách mạng tháng Tám năm 1945":
    - Connected to: "Hồ Chí Minh" (LEADER)
    - Connected to: "Việt Minh" (LED_BY)
    - Connected to: "Nhật Bản đầu hàng" (HAPPENED_AFTER)

Total expanded: 6 entities

┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Hybrid Scoring                                     │
└─────────────────────────────────────────────────────────────┘
For "Cách mạng Tháng Tám":
  - Vector score: 0.904 × 0.35 = 0.316
  - Graph score: 1.0 × 0.25 = 0.250 (distance=0)
  - Type score: 0.8 × 0.25 = 0.200 (EVENT for WHY)
  - Source score: 1.0 × 0.15 = 0.150
  → TOTAL: 0.916

Ranked top 10 entities with hybrid scores

┌─────────────────────────────────────────────────────────────┐
│ Phase 5: Extract Relationships                              │
└─────────────────────────────────────────────────────────────┘
Find relationships between selected entities:
  - "Cách mạng Tháng Tám" --[THÀNH_CÔNG]--> "Việt Nam Dân chủ Cộng hòa"
  - "Hồ Chí Minh" --[LÃNH_ĐẠO]--> "Cách mạng Tháng Tám"
  - "1945" --[THỜI_ĐIỂM]--> "Cách mạng Tháng Tám"

Total: 4 relationships

┌─────────────────────────────────────────────────────────────┐
│ Phase 6: Extract Page Numbers & Load Content               │
└─────────────────────────────────────────────────────────────┘
From entities, extract pages:
  - "Cách mạng Tháng Tám": first_seen_page = 56
  - "1945": first_seen_page = 14
  - "Việt Nam Dân chủ Cộng hòa": first_seen_page = 56

Load content from data/chapter10.md:
  → Page 56: "Tuần lễ vàng (từ ngày 16-9-1945)..."
  → Page 14: "Viện Sử học thuộc Viện Hàn lâm..."

Total sources: 5 pages with content

┌─────────────────────────────────────────────────────────────┐
│ Phase 7: Build Structured Context                          │
└─────────────────────────────────────────────────────────────┘
Context structure:
{
  "question": "Cách mạng tháng Tám...",
  "question_type": "DEFAULT",
  "entities": [
    {name: "Cách mạng Tháng Tám", type: "EVENT", score: 0.916},
    ...
  ],
  "relationships": [
    {source: "Cách mạng", target: "VNCH", type: "THÀNH_CÔNG"},
    ...
  ],
  "sources": [
    {
      "chapter": 10,
      "page": 56,
      "citation": "Chapter 10, Page 56",
      "content": "Tuần lễ vàng (từ ngày 16-9-1945)..."
    },
    ...
  ]
}

┌─────────────────────────────────────────────────────────────┐
│ Phase 8: Format Prompt for Gemini                          │
└─────────────────────────────────────────────────────────────┘
Prompt structure (Vietnamese):
  1. System instructions
  2. ENTITIES: 7 entities with scores
  3. RELATIONSHIPS: 4 relationships
  4. SOURCE CONTENT: 5 pages with actual text from book
  5. QUESTION
  6. Request for answer

Total: ~384 prompt tokens

┌─────────────────────────────────────────────────────────────┐
│ Phase 9: Generate with Gemini                              │
└─────────────────────────────────────────────────────────────┘
→ Send prompt to Gemini API
→ Generation config:
  - temperature: 0.7
  - max_output_tokens: 8192
  - model: gemini-2.5-flash

→ Generate answer...
→ Time: ~23 seconds
→ Output: 1836 tokens
→ Finish reason: STOP (complete)

┌─────────────────────────────────────────────────────────────┐
│ Phase 10: Return Structured Result                         │
└─────────────────────────────────────────────────────────────┘
{
  "question": "...",
  "answer": "Chào bạn, với vai trò là chuyên gia...",
  "context": {...},
  "metadata": {
    "retrieval_time": 5.83s,
    "generation_time": 23.27s,
    "total_time": 29.10s,
    "entities_used": 7,
    "relationships_used": 4,
    "prompt_tokens": 384,
    "completion_tokens": 1836
  }
}
```

---

## 🎯 Điểm Mạnh của Hệ Thống

### 1. **Hybrid Retrieval**
- Kết hợp semantic search (vector) với graph traversal
- Tận dụng cả similarity và structure relationships
- Question-aware scoring dựa trên question type

### 2. **Source Context Integration**
- Không chỉ dựa vào entities/relationships
- Load nội dung thực tế từ sách gốc (chapter10.md)
- LLM có access đến raw text, tăng độ chính xác

### 3. **Intelligent Ranking**
- 4 yếu tố scoring:
  - Vector similarity (35%): Semantic relevance
  - Graph proximity (25%): Structural importance  
  - Entity type (25%): Question-type matching
  - Seed quality (15%): Confidence from vector search

### 4. **Multi-level Context**
- Entities: What/Who
- Relationships: How they connect
- Paths: Indirect connections
- Source content: Actual evidence

---

## 📊 Performance Metrics

**Typical Query:**
- Retrieval time: ~5-7 seconds
  - Vector search: ~1s
  - Graph expansion: ~2s
  - Ranking: ~1s
  - Page loading: ~2s
  
- Generation time: ~20-25 seconds (Gemini API)

- Total: ~30 seconds per query

**Token Usage:**
- Average prompt: 300-500 tokens
- Average completion: 1500-2500 tokens
- Max output: 8192 tokens

---

## 🔧 Configuration

### Neo4j Graph Schema:
```cypher
// Nodes
(:PERSON {name, description, first_seen_page, first_seen_chapter})
(:LOCATION {name, description, first_seen_page, first_seen_chapter})
(:ORGANIZATION {name, description, first_seen_page, first_seen_chapter})
(:EVENT {name, description, first_seen_page, first_seen_chapter})
(:TIME {name, description, first_seen_page, first_seen_chapter})

// Relationships
()-[r:LÃNH_ĐẠO {description}]->()
()-[r:THÀNH_CÔNG {description}]->()
()-[r:THAM_GIA {description}]->()
// ... more relationship types
```

### Embedding Model:
```python
model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
dimension = 768
```

### Gemini Configuration:
```python
model = "gemini-2.5-flash"
temperature = 0.7
max_output_tokens = 8192
```

---

## 🚀 Cách Sử Dụng

### Basic Usage:
```python
from graph_rag_gemini import GeminiRAG
from graph_rag_hybrid import HybridRetriever
from graph_rag_context import ContextBuilder
from graph_rag_embeddings import EntityEmbeddings
from graph_rag_query import GraphRAGQuery

# Initialize components
graph_query = GraphRAGQuery(neo4j_graph)
embeddings = EntityEmbeddings(neo4j_graph)
hybrid = HybridRetriever(graph_query, embeddings)
builder = ContextBuilder(source_file="data/chapter10.md")

# Create RAG system
rag = GeminiRAG(
    hybrid_retriever=hybrid,
    context_builder=builder,
    model_name="gemini-2.5-flash"
)

# Ask question
result = rag.generate_answer(
    question="Cách mạng tháng Tám thành công do đâu?",
    prompt_type="explain"
)

print(result['answer'])
```

---

## 🐛 Các Vấn Đề Đã Fix

### 1. **Page Info Không Được Truyền**
**Vấn đề:** Entities trong Neo4j có `first_seen_page` nhưng không xuất hiện trong retrieval results.

**Nguyên nhân:** 
- Query trong `graph_rag_embeddings.py` không SELECT page fields
- Query trong `graph_rag_query.py` không return page info
- `graph_rag_hybrid.py` không copy page info khi build entity dicts

**Giải pháp:**
- ✅ Update tất cả Cypher queries để return `first_seen_page`, `first_seen_chapter`
- ✅ Update 3 chỗ trong hybrid.py khi build entity dicts
- ✅ Implement `_load_page_content()` trong context builder

### 2. **Gemini Response Bị Truncate**
**Vấn đề:** Câu trả lời bị cắt ngang, không hoàn chỉnh.

**Nguyên nhân:** 
- `max_output_tokens` trong generation_config quá thấp (2048)
- Finish reason = MAX_TOKENS

**Giải pháp:**
- ✅ Tăng default `max_output_tokens` lên 8192
- ✅ Add warning khi response incomplete
- ✅ Log finish reason để debug

---

## 📝 Notes

- System cần Neo4j database đã được populate với data
- Embeddings cần được generate trước (hoặc load từ database)
- Source file `data/chapter10.md` phải có format đúng với `--- Page X ---` markers
- Gemini API key cần được set trong environment variables

---

## 🎓 Tài Liệu Tham Khảo

1. **Graph RAG Papers:**
   - "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
   - Microsoft GraphRAG

2. **Technologies:**
   - Neo4j Graph Database
   - Sentence Transformers
   - Google Gemini API
   - LangChain Community

3. **Vietnamese NLP:**
   - Multilingual embedding models
   - Vietnamese tokenization challenges

---

**Tác giả:** Graph RAG System for Vietnamese History  
**Version:** 1.0  
**Last Updated:** December 25, 2025
