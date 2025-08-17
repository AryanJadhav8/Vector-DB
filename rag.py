"""
RAG Example: Cooking Assistant
Very simple pipeline showing how RAG works with a small usecase.
"""

# 1. Imports
from openai import OpenAI
import chromadb

# 2. Initialize clients
client = OpenAI()
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="cooking_guide")

# 3. Our knowledge base (tiny "cooking book")
documents = [
    "To cook pasta, boil water, add salt, then put pasta until soft.",
    "To make pizza, prepare dough, add sauce and cheese, bake in oven.",
    "To boil eggs, place eggs in water, boil for 10 minutes."
]

# Insert docs into vector DB with embeddings
for i, doc in enumerate(documents):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding

    collection.add(
        documents=[doc],
        embeddings=[embedding],
        ids=[f"doc_{i}"]
    )

# 4. User query
query = "How do I cook pasta?"

# Convert query → embedding
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# 5. Retrieve relevant docs
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

retrieved_docs = results["documents"][0]

# 6. Build final prompt with context
context = "\n".join(retrieved_docs)
prompt = f"""
You are a helpful cooking assistant.

Context:
{context}

Question: {query}

Answer in simple steps:
"""

# 7. Ask the LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

# 8. Output
print("User Question:", query)
print("Retrieved Docs:", retrieved_docs)
print("LLM Answer:", response.choices[0].message.content)
