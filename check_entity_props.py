"""Check entity properties in Neo4j"""

from langchain_community.graphs import Neo4jGraph

# Setup Neo4j
NEO4J_URI = "neo4j+s://0c367113.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "gTO1K567hBLzkRdUAhhEb-UqvBjz0i3ckV3M9v_-Nio"

graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

# First, check what labels exist
query = """
CALL db.labels()
"""
result = graph.query(query)
print("Available labels:")
for row in result:
    print(f"  {row}")

# Get all node types and their properties
query2 = """
MATCH (n)
RETURN DISTINCT labels(n) as labels, count(*) as count
LIMIT 20
"""
result2 = graph.query(query2)
print("\n\nNode labels and counts:")
for row in result2:
    print(f"  {row['labels']}: {row['count']} nodes")

# Sample a few nodes
query3 = """
MATCH (n)
WHERE n.name IS NOT NULL
RETURN labels(n) as labels, properties(n) as props
LIMIT 5
"""
result3 = graph.query(query3)
print("\n\nSample nodes with properties:")
for i, row in enumerate(result3, 1):
    print(f"\n{i}. Labels: {row['labels']}")
    print(f"   Props: {row['props']}")

