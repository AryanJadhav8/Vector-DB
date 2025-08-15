"""
Sentence → Embedding → Store in Pinecone
This script:
1. Reads sentences (list)
2. Generates embeddings using OpenAI
3. Stores embeddings in Pinecone
"""

import os
from dotenv import load_dotenv
import openai
import pinecone

# 1️ Load API keys from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENVIRONMENT")

# 2️ Initialize Pinecone
pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

# 3️ Create index if not exists
index_name = "sentence-index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)  # 1536 for OpenAI embeddings

index = pinecone.Index(index_name)

# 4️ Example sentences
sentences = [
    "AI will change the world.",
    "Pinecone is great for vector search.",
    "I love learning new technologies."
]

# 5️ Convert each sentence into embedding
for i, sentence in enumerate(sentences):
    # Get embedding from OpenAI
    embedding_response = openai.Embedding.create(
        input=sentence,
        model="text-embedding-ada-002"
    )
    vector = embedding_response['data'][0]['embedding']

    # Store in Pinecone (id must be unique)
    index.upsert([(f"id-{i}", vector, {"text": sentence})])

print(" Sentences stored in Pinecone!")
