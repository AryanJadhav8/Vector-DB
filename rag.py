"""
RAG Example: Cooking Assistant with Pinecone
Shows a minimal RAG pipeline using Pinecone as vector DB.
"""

# 1. Imports
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# 2. Initialize OpenAI client
client = OpenAI()

# 3. Initialize Pinecone
pc = Pinecone(api_key="YOUR_PINECONE_API_KEY")

index_name = "cooking-assistant"

# Create index if it doesn’t exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # dimension for text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# 4. Our knowledge base (tiny cooking book)
documents = [
    "To cook pasta, boil water, add salt, then put pasta until soft.",
    "To make pizza, prepare dough, add sauce and cheese, bake in oven.",
    "To boil eggs, place eggs in water, boil for 10 minutes."
]

# 5. Insert docs into Pinecone with embeddings
for i, doc in enumerate(documents):
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding

    index.upsert([
        {"id": f"doc_{i}", "values": embedding, "metadata": {"text": doc}}
    ])

# 6. User query
query = "How do I cook pasta?"

# Convert query → embedding
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding

# 7. Query Pinecone
results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)

retrieved_docs = [match["metadata"]["text"] for match in results["matches"]]

# 8. Build final prompt with retrieved context
context = "\n".join(retrieved_docs)
prompt = f"""
You are a helpful cooking assistant.    

Context:
{context}

Question: {query}

Answer in simple steps:
"""

# 9. Ask the LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

# 10. Output
print("User Question:", query)
print("Retrieved Docs:", retrieved_docs)
print("LLM Answer:", response.choices[0].message.content)
