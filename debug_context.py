"""Debug script để kiểm tra page content trong context"""

import os
from langchain_community.graphs import Neo4jGraph
from graph_rag_query import GraphRAGQuery
from graph_rag_embeddings import EntityEmbeddings
from graph_rag_hybrid import HybridRetriever
from graph_rag_context import ContextBuilder

# Setup Neo4j
NEO4J_URI = "neo4j+s://0c367113.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "gTO1K567hBLzkRdUAhhEb-UqvBjz0i3ckV3M9v_-Nio"

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# Initialize components
graph_query = GraphRAGQuery(graph)
embeddings = EntityEmbeddings(graph)
hybrid = HybridRetriever(graph_query, embeddings)
builder = ContextBuilder(source_file="data/chapter10.md")

# Test retrieval
question = "Liên khu Việt Bắc ở đâu?"
print(f"Question: {question}\n")

# Step 1: Check retrieval results
print("=" * 70)
print("STEP 1: RETRIEVAL RESULTS")
print("=" * 70)

retrieval_context = hybrid.retrieve(
    question=question,
    top_k=10,
    vector_top_k=5,
    expansion_depth=1
)

print(f"\nTop entities found: {len(retrieval_context['top_entities'])}")
for i, entity in enumerate(retrieval_context['top_entities'][:3], 1):
    print(f"\n{i}. {entity['name']} ({entity['type']})")
    print(f"   Score: {entity['score']:.4f}")
    print(f"   ALL KEYS: {list(entity.keys())}")
    print(f"   FULL DICT: {entity}")

# Step 2: Build context
print("\n" + "=" * 70)
print("STEP 2: CONTEXT BUILDING")
print("=" * 70)

context = builder.build_rag_context(
    question=question,
    retrieval_context=retrieval_context,
    include_sources=True
)

print(f"\nSources found: {len(context['sources'])}")
for i, source in enumerate(context['sources'], 1):
    print(f"\n{i}. {source['citation']}")
    if 'content' in source:
        content_preview = source['content'][:300].replace('\n', ' ')
        print(f"   Content length: {len(source['content'])} chars")
        print(f"   Content preview: {content_preview}...")

# Step 3: Format prompt (chỉ lấy phần có source content)
print("\n" + "=" * 70)
print("STEP 3: CHECK PROMPT SOURCE SECTION")
print("=" * 70)

prompt = builder.format_for_gemini(context, prompt_type="qa")

# Extract SOURCE CONTENT section
if "SOURCE CONTENT" in prompt:
    start_idx = prompt.find("SOURCE CONTENT")
    end_idx = prompt.find("=" * 70, start_idx + 1)
    source_section = prompt[start_idx:end_idx] if end_idx != -1 else prompt[start_idx:]
    print("\n" + source_section[:2000])  # Show first 2000 chars
else:
    print("❌ NO SOURCE CONTENT SECTION FOUND IN PROMPT!")
    print("\nPrompt sections found:")
    for line in prompt.split('\n')[:30]:
        if line.strip():
            print(f"  {line[:80]}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"✓ Entities retrieved: {len(retrieval_context['top_entities'])}")
print(f"✓ Sources extracted: {len(context['sources'])}")
print(f"✓ Prompt length: {len(prompt)} chars")
print(f"✓ SOURCE CONTENT in prompt: {'SOURCE CONTENT' in prompt}")

