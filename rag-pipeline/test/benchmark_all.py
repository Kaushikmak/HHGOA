import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunking import stream_and_chunk_parquet

# standard codes
LANG_MAP = {
    "asm": "as", "ben": "bn", "guj": "gu", "hin": "hi", "kan": "kn",
    "mal": "ml", "mar": "mr", "nep": "ne", "ori": "or", "pan": "pa",
    "san": "sa", "tam": "ta", "urd": "ur"
}

def setup_qdrant_and_embedder():
    embedder = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    client = QdrantClient(":memory:")
    return client, embedder

def evaluate_retrieval(client, embedder, collection_name, queries_to_test, eval_lang):
    hits_at_1, hits_at_3, mrr_sum = 0, 0, 0.0
    total = len(queries_to_test)
    latencies = []
    
    for q_id, q_dict in queries_to_test.items():
        query_text = q_dict.get(eval_lang)
        if not query_text:
            continue
            
        start_time = time.time()
        query_vector = embedder.encode(query_text).tolist()
        
        search_filter = Filter(must=[FieldCondition(key="lang", match=MatchValue(value=eval_lang))])
        results = client.query_points(
            collection_name=collection_name, query=query_vector, query_filter=search_filter, limit=3
        )
        latencies.append((time.time() - start_time) * 1000)
        
        found_at = -1
        for rank, res in enumerate(results.points):
            if res.payload["query_id"] == q_id and res.payload["is_selected"] is True:
                found_at = rank + 1
                break
                
        if found_at == 1: hits_at_1 += 1
        if found_at > 0 and found_at <= 3: hits_at_3 += 1
        if found_at > 0: mrr_sum += (1.0 / found_at)
            
    if not latencies: return None
    
    return {
        "Recall@1": round((hits_at_1 / total) * 100, 2),
        "Recall@3": round((hits_at_3 / total) * 100, 2),
        "MRR": round(mrr_sum / total, 4),
        "Avg_Latency": round(np.mean(latencies), 2),
        "P50": round(np.percentile(latencies, 50), 2),
        "P75": round(np.percentile(latencies, 75), 2),
        "P99": round(np.percentile(latencies, 99), 2),
        "P100": round(np.max(latencies), 2)
    }

def run_all_benchmarks(data_dir: str, num_queries: int = 50):
    client, embedder = setup_qdrant_and_embedder()
    results_list = []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('train.parquet')]
    for file in files:
        prefix = file[:3]
        lang_code = LANG_MAP.get(prefix)
        if not lang_code: continue
        
        file_path = os.path.join(data_dir, file)
        collection_name = f"benchmark_{lang_code}"
        
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedder.get_embedding_dimension(), distance=Distance.COSINE),
            )
            
        print(f"\nProcessing {lang_code.upper()} from {file}...")
        queries_to_test = {}
        points = []
        point_id = 1
        
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(file_path)
        df = next(pf.iter_batches(batch_size=num_queries)).to_pandas()
        
        for _, row in df.iterrows():
            queries_to_test[str(row["query_id"])] = {
                "en": row.get("Eng_Query", ""),
                lang_code: row.get("query", "")
            }

        for batch_nodes in stream_and_chunk_parquet(file_path, lang_code=lang_code, batch_size=num_queries):
            texts = [node.text for node in batch_nodes]
            embeddings = embedder.encode(texts, show_progress_bar=False)
            
            for i, node in enumerate(batch_nodes):
                points.append(PointStruct(
                    id=point_id, vector=embeddings[i].tolist(),
                    payload={
                        "text": node.text, "query_id": node.metadata["query_id"],
                        "is_selected": node.metadata["is_selected"], "lang": node.metadata["lang"]
                    }
                ))
                point_id += 1
            break # Just one batch
            
        client.upsert(collection_name=collection_name, points=points)
        
        # Evaluate Target Language
        tgt_metrics = evaluate_retrieval(client, embedder, collection_name, queries_to_test, eval_lang=lang_code)
        if tgt_metrics:
            tgt_metrics["Language"] = lang_code.upper()
            tgt_metrics["Corpus"] = "Target"
            results_list.append(tgt_metrics)
            
        # Evaluate English
        en_metrics = evaluate_retrieval(client, embedder, collection_name, queries_to_test, eval_lang="en")
        if en_metrics:
            en_metrics["Language"] = lang_code.upper()
            en_metrics["Corpus"] = "English"
            results_list.append(en_metrics)
            
    # Compile Results
    df_res = pd.DataFrame(results_list)
    output_dir = "rag-pipeline/test/test_result"
    os.makedirs(output_dir, exist_ok=True)
    
    df_res.to_csv(os.path.join(output_dir, "benchmark_results.csv"), index=False)
    print(f"\nBenchmarking complete. Results saved to {output_dir}/benchmark_results.csv")
    
    # Generate Combined Plots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("IndicRAG Benchmark Results across 13 Languages", fontsize=16)
    
    # Subplot 1: Recall@1 Comparison
    sns.barplot(data=df_res, x="Language", y="Recall@1", hue="Corpus", ax=axes[0, 0])
    axes[0, 0].set_title("Recall@1 (%)")
    axes[0, 0].set_ylim(0, 100)
    
    # Subplot 2: Recall@3 Comparison
    sns.barplot(data=df_res, x="Language", y="Recall@3", hue="Corpus", ax=axes[0, 1])
    axes[0, 1].set_title("Recall@3 (%)")
    axes[0, 1].set_ylim(0, 100)
    
    # Subplot 3: MRR Comparison
    sns.barplot(data=df_res, x="Language", y="MRR", hue="Corpus", ax=axes[1, 0])
    axes[1, 0].set_title("Mean Reciprocal Rank (MRR)")
    axes[1, 0].set_ylim(0, 1.0)
    
    # Subplot 4: P99 Latency (Target Languages Only)
    sns.barplot(data=df_res[df_res["Corpus"] == "Target"], x="Language", y="P99", hue="Language", palette="viridis", legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("P99 Vector Search Latency (ms)")
    axes[1, 1].set_ylabel("Latency (ms)")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_benchmark_results.png"), dpi=300, bbox_inches='tight')
    print(f"Combined benchmark plots saved to {output_dir}/combined_benchmark_results.png")

if __name__ == "__main__":
    run_all_benchmarks("data/train/", num_queries=50)
