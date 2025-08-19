"""
pip3 install langchain langchain_community langchain_core langchain_openai langchain_mongodb pymongo pypdf
Create a key_param file with the following content:
"""

MONGODB_URI=<your_atlas_connection_string>
LLM_API_KEY=<your_llm_api_key>

"""
load_data.py file

This code ingests a pdf, removes empty pages, chunks the pages into paragraphs, generates metadata for the chunks, creates embeddings, and stores the chunks and their embeddings in a MongoDB collection.
"""
#import libraries
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)

import key_param

# Set the MongoDB URI, DB, Collection Names

# Import MongoDB client and connect using your URI
client = MongoClient(key_param.MONGODB_URI)

# Define database and collection where chunked documents + embeddings will be stored
dbName = "book_mongodb_chunks"
collectionName = "chunked_data"
collection = client[dbName][collectionName]


# 1. LOAD THE PDF DOCUMENT
# Use PyPDFLoader to read the PDF into LangChain "Document" objects (page by page)
loader = PyPDFLoader("./sample_files/mongodb.pdf")
pages = loader.load()

# Clean pages: only keep pages with more than 20 words (remove blank/short pages)
cleaned_pages = []
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)


# 2. SPLIT INTO CHUNKS
# RecursiveCharacterTextSplitter splits text into chunks of max 500 characters
# with 150 characters overlapping between chunks (to preserve context).
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)


# 3. DEFINE METADATA SCHEMA
# Metadata will enrich each document with structured info.
# Example schema: title (string), keywords (list of strings), hasCode (boolean).
schema = {
    "properties": {
        "title": {"type": "string"},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "hasCode": {"type": "boolean"},
    },
    "required": ["title", "keywords", "hasCode"],  # must be included
}


# 4. METADATA TAGGING WITH LLM
# Create an OpenAI LLM (ChatGPT-3.5-turbo) that will analyze documents
# and automatically extract metadata according to schema above.
llm = ChatOpenAI(
    openai_api_key=key_param.LLM_API_KEY,
    temperature=0,  # deterministic output
    model="gpt-3.5-turbo"
)

# Build metadata transformer using the schema + LLM
document_transformer = create_metadata_tagger(metadata_schema=schema, llm=llm)

# Transform documents: add metadata to each page (title, keywords, hasCode, etc.)
docs = document_transformer.transform_documents(cleaned_pages)


# 5. APPLY TEXT SPLITTING
# Split enriched documents into smaller chunks for embeddings
split_docs = text_splitter.split_documents(docs)


# 6. CREATE EMBEDDINGS
# Create embeddings for chunks using OpenAI's embedding model
embeddings = OpenAIEmbeddings(openai_api_key=key_param.LLM_API_KEY)


# 7. STORE IN MONGODB VECTOR SEARCH
# Push chunks + embeddings into MongoDB Atlas Vector Search collection
# This allows you to later perform semantic search over the PDF content
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    split_docs,
    embeddings,
    collection=collection
)
