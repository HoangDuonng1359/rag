# Các Công Thức Toán Học Được Sử Dụng Trong Project RAG

## Tổng quan

Document này tổng hợp tất cả các công thức toán học được sử dụng trong hệ thống RAG (Retrieval-Augmented Generation) bao gồm Traditional RAG và Graph RAG.

---

## 1. Vector Embeddings & Similarity

### 1.1. Cosine Similarity

**Định nghĩa:** Đo độ tương đồng giữa hai vector dựa trên góc giữa chúng.

**Công thức:**

$$\text{cosine\_similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \times \|\mathbf{b}\|}$$

Trong đó:
- $\mathbf{a}, \mathbf{b}$: Hai vector embeddings (768 chiều)
- $\mathbf{a} \cdot \mathbf{b}$: Tích vô hướng (dot product) = $\sum_{i=1}^{n} a_i \times b_i$
- $\|\mathbf{a}\|$: Chuẩn Euclidean (norm) = $\sqrt{\sum_{i=1}^{n} a_i^2}$

**Giá trị:**
- Range: $[-1, 1]$
- $1$: Hoàn toàn giống nhau
- $0$: Độc lập (vuông góc)
- $-1$: Hoàn toàn đối lập

**Sử dụng trong code:**
- File: `graph_rag_embeddings.py` (line 283)
- File: `chunk.py` (line 58)

```python
# graph_rag_embeddings.py
def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# chunk.py
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(vectors[i:i+1], vectors[i+1:i+2])[0][0]
```

---

### 1.2. Cosine Distance

**Định nghĩa:** Khoảng cách cosine, sử dụng trong ChromaDB.

**Công thức:**

$$\text{cosine\_distance}(\mathbf{a}, \mathbf{b}) = 1 - \text{cosine\_similarity}(\mathbf{a}, \mathbf{b})$$

**Chuyển đổi sang similarity:**

$$\text{similarity} = 1 - \text{distance}$$

**Giá trị:**
- Range: $[0, 2]$
- $0$: Hoàn toàn giống nhau
- $2$: Hoàn toàn đối lập

**Sử dụng trong code:**
- File: `traditional_rag_vectorstore.py` (line 196-197)

```python
# ChromaDB trả về distance
distance = results['distances'][0][i]
# Convert sang similarity
similarity = 1.0 - distance
```

---

### 1.3. Vector Normalization

**Định nghĩa:** Chuẩn hóa vector về unit length (độ dài = 1).

**Công thức:**

$$\mathbf{v}_{\text{normalized}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

Trong đó:
- $\mathbf{v}$: Vector gốc
- $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2}$: Chuẩn Euclidean

**Tính chất:**
- $\|\mathbf{v}_{\text{normalized}}\| = 1$
- Sau khi normalize, cosine similarity = dot product

**Sử dụng trong code:**
- File: `traditional_rag_embeddings.py` (line 48-50, 78-79)

```python
embedding = self.model.encode(
    text,
    normalize_embeddings=True,  # Normalize về unit length
    convert_to_numpy=True
)
```

---

## 2. Graph RAG - Hybrid Scoring

### 2.1. Hybrid Score Formula

**Định nghĩa:** Điểm số kết hợp từ nhiều signals để rank entities trong Graph RAG.

**Công thức:**

$$\text{hybrid\_score} = w_1 \cdot s_{\text{vector}} + w_2 \cdot s_{\text{graph}} + w_3 \cdot s_{\text{type}} + w_4 \cdot s_{\text{source}}$$

Trong đó:
- $s_{\text{vector}}$: Vector similarity score (semantic matching)
- $s_{\text{graph}}$: Graph proximity score (khoảng cách trong đồ thị)
- $s_{\text{type}}$: Entity type relevance score
- $s_{\text{source}}$: Seed entity quality score

**Weights (hệ số):**
- $w_1 = 0.35$ (35%) - Direct semantic match
- $w_2 = 0.25$ (25%) - Graph proximity
- $w_3 = 0.25$ (25%) - Type relevance
- $w_4 = 0.15$ (15%) - Seed quality

**Ràng buộc:**

$$\sum_{i=1}^{4} w_i = 1.0$$

**Sử dụng trong code:**
- File: `graph_rag_hybrid.py` (line 262-274)

```python
hybrid_score = (
    0.35 * vector_score +      # Direct semantic match
    0.25 * graph_score +        # Graph proximity
    0.25 * type_score +         # Type relevance
    0.15 * source_score        # Seed quality
)
```

---

### 2.2. Graph Proximity Score

**Định nghĩa:** Điểm số dựa trên khoảng cách trong knowledge graph.

**Công thức:**

$$s_{\text{graph}} = \frac{1}{d + 1}$$

Trong đó:
- $d$: Shortest path distance (số bước từ seed entity)
- $d \in [0, 1, 2, 3, ...]$

**Giá trị:**
- $d = 0$: $s_{\text{graph}} = 1.0$ (chính seed entity)
- $d = 1$: $s_{\text{graph}} = 0.5$ (1-hop neighbor)
- $d = 2$: $s_{\text{graph}} = 0.33$ (2-hop neighbor)
- $d = 3$: $s_{\text{graph}} = 0.25$ (3-hop neighbor)

**Sử dụng trong code:**
- File: `graph_rag_hybrid.py` (line 256-257)

```python
distance = entity.get('distance', 3)
graph_score = 1.0 / (distance + 1)
```

---

### 2.3. Entity Type Matching Score

**Định nghĩa:** Điểm số dựa trên mức độ phù hợp của entity type với question type.

**Công thức:**

$$s_{\text{type}} = w_{\text{q\_type}, \text{e\_type}}$$

**Weight Matrix:**

| Question Type | PERSON | LOCATION | ORGANIZATION | EVENT | TIME |
|---------------|--------|----------|--------------|-------|------|
| WHO | 1.0 | - | 0.7 | 0.3 | - |
| WHERE | - | 1.0 | 0.3 | 0.5 | - |
| WHEN | 0.2 | - | - | 0.8 | 1.0 |
| WHAT | 0.4 | - | 0.6 | 1.0 | - |
| DEFAULT | 0.5 | 0.5 | 0.5 | 0.8 | 0.5 |

**Sử dụng trong code:**
- File: `graph_rag_hybrid.py` (line 41-48, 259-260)

```python
type_weights = {
    'WHO': {'PERSON': 1.0, 'ORGANIZATION': 0.7, 'EVENT': 0.3},
    'WHERE': {'LOCATION': 1.0, 'EVENT': 0.5, 'ORGANIZATION': 0.3},
    'WHEN': {'TIME': 1.0, 'EVENT': 0.8, 'PERSON': 0.2},
    'WHAT': {'EVENT': 1.0, 'ORGANIZATION': 0.6, 'PERSON': 0.4},
    'DEFAULT': {'PERSON': 0.5, 'LOCATION': 0.5, 'ORGANIZATION': 0.5, 
               'EVENT': 0.8, 'TIME': 0.5}
}

entity_type = entity['type']
type_score = type_weights.get(entity_type, 0.3)
```

---

## 3. Document Chunking - Semantic Breakpoints

### 3.1. Breakpoint Score

**Định nghĩa:** Điểm số để xác định điểm chia tốt nhất cho document chunking.

**Công thức:**

$$\text{score}(i) = (1 - \text{sim}(i)) \times 100 - |\text{cumsum}(i) - \text{target}|$$

Trong đó:
- $\text{sim}(i)$: Cosine similarity giữa câu $i$ và câu $i+1$
- $\text{cumsum}(i)$: Tổng số từ tích lũy đến câu $i$
- $\text{target}$: Số từ mục tiêu cho mỗi chunk

**Ý nghĩa:**
- $(1 - \text{sim}(i))$: Ưu tiên điểm có similarity thấp (topic thay đổi)
- $|\text{cumsum}(i) - \text{target}|$: Penalize điểm xa target length

**Sử dụng trong code:**
- File: `chunk.py` (line 68-71)

```python
# Ưu tiên điểm có similarity thấp và gần target
if cumsum[i] >= self.min_words and cumsum[i] <= self.max_words * 1.2:
    score = (1 - sim) * 100 - abs(cumsum[i] - target_words)
    breakpoints.append((i+1, score))
```

---

## 4. TF-IDF (Term Frequency-Inverse Document Frequency)

### 4.1. TF (Term Frequency)

**Công thức:**

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

Trong đó:
- $f_{t,d}$: Số lần term $t$ xuất hiện trong document $d$
- $\sum_{t' \in d} f_{t',d}$: Tổng số terms trong document $d$

---

### 4.2. IDF (Inverse Document Frequency)

**Công thức:**

$$\text{IDF}(t, D) = \log \frac{N}{|\{d \in D : t \in d\}|}$$

Trong đó:
- $N$: Tổng số documents
- $|\{d \in D : t \in d\}|$: Số documents chứa term $t$

---

### 4.3. TF-IDF Score

**Công thức:**

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

**Sử dụng trong code:**
- File: `chunk.py` (line 52-54)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=100)
vectors = vectorizer.fit_transform(sentences)
```

**Ý nghĩa:**
- High TF-IDF: Term quan trọng và đặc trưng cho document
- Low TF-IDF: Term phổ biến hoặc ít xuất hiện

---

## 5. Evaluation Metrics

### 5.1. Precision@K

**Định nghĩa:** Tỷ lệ documents liên quan trong top-K results.

**Công thức:**

$$\text{Precision@K} = \frac{\text{relevant\_in\_top\_k}}{k}$$

**Range:** $[0, 1]$
- $1$: Tất cả top-K đều relevant
- $0$: Không có relevant document nào

---

### 5.2. Recall@K

**Định nghĩa:** Tỷ lệ relevant documents được retrieve trong top-K.

**Công thức:**

$$\text{Recall@K} = \frac{\text{relevant\_in\_top\_k}}{\text{total\_relevant}}$$

**Range:** $[0, 1]$
- $1$: Tìm được tất cả relevant documents
- $0$: Không tìm được relevant document nào

---

### 5.3. MRR (Mean Reciprocal Rank)

**Định nghĩa:** Trung bình nghịch đảo của rank của kết quả đúng đầu tiên.

**Công thức:**

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

Trong đó:
- $|Q|$: Số lượng queries
- $\text{rank}_i$: Position của relevant result đầu tiên cho query $i$

**Ví dụ:**
- Relevant ở vị trí 1: $\text{RR} = 1/1 = 1.0$
- Relevant ở vị trí 2: $\text{RR} = 1/2 = 0.5$
- Relevant ở vị trí 3: $\text{RR} = 1/3 = 0.33$

**Sử dụng:** Đánh giá chất lượng ranking của retrieval system

---

## 6. Distance Metrics

### 6.1. Euclidean Distance

**Công thức:**

$$d_{\text{euclidean}}(\mathbf{a}, \mathbf{b}) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}$$

**Ý nghĩa:** Khoảng cách đường thẳng giữa hai điểm trong không gian n-chiều.

---

### 6.2. Manhattan Distance (L1 Distance)

**Công thức:**

$$d_{\text{manhattan}}(\mathbf{a}, \mathbf{b}) = \sum_{i=1}^{n} |a_i - b_i|$$

**Ý nghĩa:** Tổng khoảng cách theo từng chiều (như đi taxi trong thành phố).

---

## 7. Neural Network Embeddings

### 7.1. Embedding Dimension

Model: `paraphrase-multilingual-mpnet-base-v2`

$$\mathbf{e} \in \mathbb{R}^{768}$$

- Input: Text string (tiếng Việt)
- Output: 768-dimensional dense vector
- Normalized: $\|\mathbf{e}\| = 1$

---

### 7.2. Batch Encoding

**Công thức:**

$$\mathbf{E} = \text{Encode}(\{\text{text}_1, \text{text}_2, ..., \text{text}_n\})$$

Trong đó:
- $\mathbf{E} \in \mathbb{R}^{n \times 768}$: Matrix của embeddings
- $n$: Số lượng texts trong batch

**Sử dụng trong code:**
```python
embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True
)
# embeddings.shape = (n, 768)
```

---

## 8. Graph Metrics

### 8.1. Shortest Path Distance

**Định nghĩa:** Số edges nhỏ nhất cần đi qua giữa hai nodes.

**Công thức (trong Neo4j Cypher):**
```cypher
MATCH path = shortestPath((a)-[*]-(b))
RETURN length(path) as distance
```

**Ý nghĩa:**
- $d = 0$: Cùng một node
- $d = 1$: Kết nối trực tiếp (1 hop)
- $d = 2$: Kết nối qua 1 node trung gian (2 hops)

---

### 8.2. Node Centrality

**Degree Centrality:**

$$C_{\text{degree}}(v) = \frac{\text{deg}(v)}{n - 1}$$

Trong đó:
- $\text{deg}(v)$: Số edges kết nối với node $v$
- $n$: Tổng số nodes

**Ý nghĩa:** Đo "importance" của node dựa trên số connections.

---

## 9. Probability & Statistics

### 9.1. Softmax (Temperature Scaling)

**Công thức:**

$$\text{softmax}(z_i) = \frac{e^{z_i/T}}{\sum_{j=1}^{n} e^{z_j/T}}$$

Trong đó:
- $z_i$: Score của item $i$
- $T$: Temperature parameter
- $T = 1$: Standard softmax
- $T > 1$: More uniform distribution
- $T < 1$: More peaked distribution

**Sử dụng:** Convert scores thành probability distribution.

---

### 9.2. Weighted Average

**Công thức:**

$$\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$

Trong đó:
- $x_i$: Giá trị thứ $i$
- $w_i$: Trọng số cho giá trị thứ $i$

---

## 10. Token Efficiency

### 10.1. Tokens per Document

**Công thức:**

$$\text{tokens} \approx \frac{\text{words} \times 1.3}{1}$$

**Ước lượng:**
- 1 word ≈ 1.3 tokens (tiếng Anh)
- 1 word ≈ 1.5-2 tokens (tiếng Việt)

---

### 10.2. Cost Calculation

**Công thức:**

$$\text{cost} = \frac{\text{input\_tokens}}{1000} \times \text{input\_price} + \frac{\text{output\_tokens}}{1000} \times \text{output\_price}$$

**Ví dụ (Gemini 2.5 Flash):**
- Input price: $0.00015 per 1K tokens
- Output price: $0.0006 per 1K tokens

---

## 11. Summary Table

| Công thức | Mục đích | File sử dụng |
|-----------|----------|--------------|
| Cosine Similarity | Đo độ tương đồng vectors | `graph_rag_embeddings.py`, `chunk.py` |
| Cosine Distance | Khoảng cách trong ChromaDB | `traditional_rag_vectorstore.py` |
| Vector Normalization | Chuẩn hóa embeddings | `traditional_rag_embeddings.py` |
| Hybrid Score | Ranking trong Graph RAG | `graph_rag_hybrid.py` |
| Graph Proximity | Điểm graph distance | `graph_rag_hybrid.py` |
| TF-IDF | Document vectorization | `chunk.py` |
| Breakpoint Score | Semantic chunking | `chunk.py` |
| Precision@K, Recall@K | Evaluation metrics | Documentation |

---

## 12. Performance Metrics

### 12.1. Latency

**Formula:**

$$\text{total\_time} = t_{\text{retrieval}} + t_{\text{context}} + t_{\text{generation}}$$

**Observed Values:**

| System | Retrieval | Context | Generation | Total |
|--------|-----------|---------|------------|-------|
| Traditional RAG | 0.22s | ~0s | 3.79s | ~4s |
| Graph RAG | 7.07s | 2-3s | 23s | ~30s |

---

### 12.2. Throughput

**Công thức:**

$$\text{throughput} = \frac{\text{queries}}{\text{time}} \text{ (queries/second)}$$

---

## Tài liệu tham khảo

### Papers
1. **Sentence-BERT**: Reimers & Gurevych (2019)
2. **RAG**: Lewis et al. (2020) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
3. **Graph RAG**: Microsoft Research (2024)

### Libraries
- **Sentence Transformers**: https://www.sbert.net/
- **ChromaDB**: https://docs.trychroma.com/
- **Neo4j**: https://neo4j.com/docs/
- **scikit-learn**: https://scikit-learn.org/

### Implementation
- **Project location**: `/home/duo/work/DL/btl/`
- **Language**: Python 3.x
- **Main models**: 
  - Embeddings: `paraphrase-multilingual-mpnet-base-v2`
  - LLM: `gemini-2.5-flash`

---

## Notes

1. **Cosine vs Euclidean:** Cosine similarity không phụ thuộc vào magnitude, chỉ phụ thuộc góc. Euclidean distance phụ thuộc cả magnitude và góc.

2. **Normalization importance:** Khi vectors được normalize về unit length, cosine similarity = dot product, giúp tính toán nhanh hơn.

3. **Hybrid scoring weights:** Weights $(0.35, 0.25, 0.25, 0.15)$ được tune empirically dựa trên performance experiments.

4. **TF-IDF vs Neural Embeddings:** TF-IDF là sparse vectors (lexical matching), neural embeddings là dense vectors (semantic matching).

5. **Graph distance:** Trong thực tế, limit expansion depth = 1-2 hops để tránh exponential growth.

---

*Document created: December 26, 2025*  
*Project: Vietnamese History RAG System*  
*Author: BTL Team*
