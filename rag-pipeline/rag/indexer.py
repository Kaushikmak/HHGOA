import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunking import stream_and_chunk_parquet

def run_indexer(data_dir: str, db_path: str, max_batches: int = 5, batch_size: int = 50):
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    
    os.makedirs(db_path, exist_ok=True)
    client = QdrantClient(path=db_path)
    
    files = [f for f in os.listdir(data_dir) if f.endswith('train.parquet')]
    
    LANG_MAP = {
        "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn",
        "mal": "ml", "mar": "mr", "nep": "ne", "ori": "or", "pan": "pa",
        "san": "sa", "tam": "ta", "urd": "ur"
    }
    
    for file in files:
        prefix = file[:3]
        lang_code = LANG_MAP.get(prefix)
        if not lang_code: 
            continue
            
        file_path = os.path.join(data_dir, file)
        collection_name = f"rag_{lang_code}"
        
        if not client.collection_exists(collection_name):
            from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType, HnswConfigDiff, PayloadSchemaType
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedder.get_embedding_dimension(), distance=Distance.COSINE),
                # Scale optimizations for production (will be utilized if deployed on real Qdrant server)
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100
                )
            )
            # Create payload indexes for faster exact-match metadata filtering
            client.create_payload_index(collection_name, "lang", field_schema=PayloadSchemaType.KEYWORD)
            client.create_payload_index(collection_name, "is_selected", field_schema=PayloadSchemaType.BOOL)
            print(f"Created collection {collection_name} with INT8 Quantization and Payload Indexes.")
            
        print(f"\nIndexing {lang_code.upper()} from {file} into '{collection_name}'...")
        
        point_id = 1
        batches_processed = 0
        
        for batch_nodes in stream_and_chunk_parquet(file_path, lang_code=lang_code, batch_size=batch_size):
            if not batch_nodes:
                continue
                
            texts = [node.text for node in batch_nodes]
            embeddings = embedder.encode(texts, show_progress_bar=False)
            
            points = []
            for i, node in enumerate(batch_nodes):
                points.append(PointStruct(
                    id=point_id, 
                    vector=embeddings[i].tolist(),
                    payload={
                        "text": node.text, 
                        "query_id": node.metadata["query_id"],
                        "is_selected": node.metadata["is_selected"], 
                        "lang": node.metadata["lang"]
                    }
                ))
                point_id += 1
            
            client.upsert(collection_name=collection_name, points=points)
            print(f"Upserted batch {batches_processed+1} ({len(points)} chunks).")
            
            batches_processed += 1
            if batches_processed >= max_batches:
                break
                
    print("\nIndexing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/train/")
    parser.add_argument("--db_path", type=str, default="../qdrant_db")
    parser.add_argument("--max_batches", type=int, default=5)
    args = parser.parse_args()
    
    run_indexer(args.data_dir, args.db_path, args.max_batches)
