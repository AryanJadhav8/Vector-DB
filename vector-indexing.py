"""
Create a Vector Search Index
To create a vector search index, use createSearchIndex() method, which expects the name, type, and definition of the index.
In this example, we use the createSearchIndex() method to create an index named vectorPlotIndex,
which is a vectorSearch index.
"""

db.movies.createSearchIndex(
  "vectorPlotIndex",
  "vectorSearch",
  {
     "fields": [
        {
           "type": "vector",
           "path": "plot_embedding",
           "numDimensions": 1536,
           "similarity": "cosine"
        }
     ]
  }
);


"""
Create a Vector Search Index with a Pre-filter Field
To create a vector search index, use createSearchIndex() method,
which expects the name, type, and definition of the index.
In this example, we use the type filter so that we can pre-filter on the year field when we use $vectorSearch.
"""
db.movies.createSearchIndex(
  "vectorPlotIndex",
  "vectorSearch",
  {
     "fields": [
        {
           "type": "vector",
           "path": "plot_embedding",
           "numDimensions": 1536,
           "similarity": "cosine"
        },
        {
          "type": "filter",
          "path": "year"
        }
     ]
  }
);

"""
Create a Vector Search Query with a Filter
The following code uses $vectorSearch with a filter on the
year field to pre-filter the data before performing a vector search.
"""

pipeline = [
    {
        "$vectorSearch": {
            "index": "vectorPlotIndex",
            "path": "plot_embedding",
            "queryVector": embedding,
            "numCandidates": 100,
            "filter": {"year": {"$gt": 2010}},
            "limit": 10
        }
    },
    {
        "$project": {
            "title": 1,
            "plot": 1,
            "year": 1,
            "score": {"$meta": "vectorSearchScore"}
        }
    }
]

x = collection.aggregate(pipeline)