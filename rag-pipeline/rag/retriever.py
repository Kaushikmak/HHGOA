import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

class VectorRetriever:
    def __init__(self, db_path: str = "qdrant_db"):
        if db_path == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_db_path = os.path.join(base_dir, db_path)
            self.client = QdrantClient(path=full_db_path)
            
        self.embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def retrieve(self, query: str, lang_code: str, top_k: int = 3):
        collection_name = f"rag_{lang_code}"
        
        if not self.client.collection_exists(collection_name):
            print(f"Warning: Collection {collection_name} does not exist.")
            return []
            
        query_vector = self.embedder.encode(query).tolist()
        
        # We only want to retrieve chunks that are in the expected language
        # or we can allow cross-language. Let's filter by language.
        search_filter = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=lang_code))])
        
        from qdrant_client.models import SearchParams
        
        # Low hnsw_ef is faster but slightly less accurate than default 128
        search_params = SearchParams(
            hnsw_ef=64, 
            exact=False  # Allow approximate search
        )
        
        results = self.client.query_points(
            collection_name=collection_name, 
            query=query_vector, 
            query_filter=search_filter, 
            search_params=search_params,
            limit=top_k
        )
        
        return [res.payload["text"] for res in results.points]
