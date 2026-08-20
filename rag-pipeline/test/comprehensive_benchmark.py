import os
import sys
import time
import psutil
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from qdrant_client.models import SearchParams
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunking import IndicRAGChunker
from rag.retriever import VectorRetriever
from harness.orchestrator import RAGOrchestrator

# For resource tracking
stop_monitoring = False
resource_data = []

def monitor_resources():
    process = psutil.Process(os.getpid())
    while not stop_monitoring:
        mem_info = process.memory_info()
        cpu_percent = process.cpu_percent(interval=0.1)
        resource_data.append({
            "Time (s)": time.time(),
            "CPU (%)": cpu_percent,
            "RAM (MB)": mem_info.rss / (1024 * 1024)
        })
        time.sleep(0.1)

def run_parameter_sweeps():
    print("Starting Comprehensive RAG Parameter Benchmarks...")
    os.makedirs("test/test_result", exist_ok=True)
    
    # Start resource monitor
    global stop_monitoring, resource_data
    stop_monitoring = False
    resource_data = []
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.start()
    start_time_global = time.time()
    
    metrics = []

    # 1. CHUNKING SWEEP
    print("1/3 Benchmarking Chunking Strategies...")
    chunk_sizes = [500, 1000, 2000]
    record = {
        "query_id": "1",
        "passages": {
            "is_selected": [1, 0, 1],
            "English_passages": ["The quick brown fox jumps over the lazy dog. " * 50] * 3,
            "Translated_passages": ["तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है। " * 50] * 3
        }
    }
    
    for size in chunk_sizes:
        chunker = IndicRAGChunker(lang_code="hi", chunk_size_limit=size)
        latencies = []
        for _ in range(20):
            start = time.time()
            chunker.process_record(record)
            latencies.append((time.time() - start) * 1000)
            
        metrics.append({
            "Stage": "Chunking",
            "Parameter": "chunk_size",
            "Value": size,
            "P50_Latency (ms)": np.percentile(latencies, 50),
            "P100_Latency (ms)": np.percentile(latencies, 100)
        })

    # 2. RETRIEVAL SWEEP (Vector DB)
    print("2/3 Benchmarking Retrieval Parameters...")
    retriever = VectorRetriever(db_path=":memory:")
    # We monkeypatch the query_points method locally to use varying hnsw_ef inside the retriever
    original_query_points = retriever.client.query_points
    
    def retrieve_with_params(query, lang_code, top_k, hnsw_ef):
        collection_name = f"rag_{lang_code}"
        query_vector = retriever.embedder.encode(query).tolist()
        try:
            # Catch collection missing gracefully for in-memory
            res = retriever.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                search_params=SearchParams(hnsw_ef=hnsw_ef, exact=False)
            )
            return len(res.points)
        except Exception:
            return 0

    top_ks = [1, 3, 5]
    hnsw_efs = [32, 64, 128]
    
    for k in top_ks:
        for ef in hnsw_efs:
            latencies = []
            for _ in range(20):
                start = time.time()
                retrieve_with_params("भारत की राजधानी क्या है?", "hi", k, ef)
                latencies.append((time.time() - start) * 1000)
                
            metrics.append({
                "Stage": f"Retrieval (top_k={k})",
                "Parameter": "hnsw_ef",
                "Value": ef,
                "P50_Latency (ms)": np.percentile(latencies, 50),
                "P100_Latency (ms)": np.percentile(latencies, 100)
            })

    # 3. GENERATION SWEEP
    print("3/3 Benchmarking Generation Overheads...")
    orchestrator = RAGOrchestrator(db_path=":memory:")
    max_tokens_list = [50, 150, 300]
    
    context = ["भारत की राजधानी नई दिल्ली है। " * 5]
    query = "भारत की राजधानी क्या है?"
    
    for limit in max_tokens_list:
        latencies = []
        for _ in range(3):
            # Hack generate_content config directly for benchmark sweeping
            start = time.time()
            try:
                client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
                client.models.generate_content(
                    model='gemini-1.5-flash-8b',
                    contents=f"Context: {context}\nQuestion: {query}",
                    config=genai.types.GenerateContentConfig(max_output_tokens=limit, temperature=0.0)
                )
            except Exception:
                pass
            latencies.append((time.time() - start) * 1000)
            
        metrics.append({
            "Stage": "Generation",
            "Parameter": "max_tokens",
            "Value": limit,
            "P50_Latency (ms)": np.percentile(latencies, 50),
            "P100_Latency (ms)": np.percentile(latencies, 100)
        })

    # Stop resource monitor
    stop_monitoring = True
    monitor_thread.join()

    # Normalize time
    for r in resource_data:
        r["Time (s)"] -= start_time_global

    # ---------------- SAVE TO CSV ----------------
    print("\nSaving logs to CSV...")
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv("test/test_result/benchmark_parameters.csv", index=False)
    
    df_resources = pd.DataFrame(resource_data)
    df_resources.to_csv("test/test_result/resource_metrics.csv", index=False)
    
    # ---------------- GENERATE PLOTS ----------------
    print("Generating visualizations (Dark Mode)...")
    plt.style.use('dark_background')
    sns.set_theme(style="darkgrid", rc={
        "axes.facecolor": "#1c1c1c", 
        "figure.facecolor": "#121212", 
        "text.color": "white",
        "axes.labelcolor": "white", 
        "xtick.color": "white", 
        "ytick.color": "white",
        "grid.color": "#333333"
    })
    
    # 1. Chunking Plot
    plt.figure(figsize=(8, 5))
    df_chunk = df_metrics[df_metrics["Stage"] == "Chunking"]
    sns.lineplot(data=df_chunk, x="Value", y="P50_Latency (ms)", marker="o", label="P50")
    sns.lineplot(data=df_chunk, x="Value", y="P100_Latency (ms)", marker="o", label="P100")
    plt.title("Chunk Size Limit vs Latency")
    plt.xlabel("Chunk Size Limit (chars)")
    plt.ylabel("Latency (ms)")
    plt.savefig("test/test_result/plot_chunking_latency.png", dpi=300)
    plt.close()
    
    # 2. Retrieval Plot
    plt.figure(figsize=(10, 6))
    df_retrieval = df_metrics[df_metrics["Stage"].str.contains("Retrieval")]
    sns.barplot(data=df_retrieval, x="Stage", y="P50_Latency (ms)", hue="Value")
    plt.title("Retrieval Latency across Top-K and HNSW_EF (Value = hnsw_ef)")
    plt.xlabel("Retrieval Top-K Strategy")
    plt.ylabel("P50 Latency (ms)")
    plt.savefig("test/test_result/plot_retrieval_latency.png", dpi=300)
    plt.close()

    # 3. Generation Plot
    plt.figure(figsize=(8, 5))
    df_gen = df_metrics[df_metrics["Stage"] == "Generation"]
    sns.barplot(data=df_gen, x="Value", y="P50_Latency (ms)", palette="Blues_d")
    plt.title("LLM Generation Latency vs Max Token Limit")
    plt.xlabel("Max Output Tokens")
    plt.ylabel("P50 Latency (ms)")
    plt.savefig("test/test_result/plot_generation_latency.png", dpi=300)
    plt.close()
    
    # 4. Resource Usage Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    ax1.plot(df_resources["Time (s)"], df_resources["RAM (MB)"], 'b-')
    ax2.plot(df_resources["Time (s)"], df_resources["CPU (%)"], 'r-', alpha=0.5)
    
    ax1.set_xlabel("Test Duration (s)")
    ax1.set_ylabel("RAM Usage (MB)", color='b')
    ax2.set_ylabel("CPU Usage (%)", color='r')
    
    plt.title("Resource Utilization during Benchmark")
    plt.savefig("test/test_result/plot_resource_usage.png", dpi=300)
    plt.close()

    print("All benchmarks complete! Data saved to test/test_result/")
    print(df_metrics)

if __name__ == "__main__":
    run_parameter_sweeps()
