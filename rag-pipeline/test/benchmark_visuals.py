import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fastapi.testclient import TestClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.chunking import IndicRAGChunker
from rag.retriever import VectorRetriever
from harness.orchestrator import RAGOrchestrator
from api.main import app

def run_benchmarks():
    print("Running comprehensive benchmarks...")
    
    # 1. Benchmark Chunking
    print("Benchmarking Chunking...")
    chunker = IndicRAGChunker(lang_code="hi", chunk_size_limit=200)
    record = {
        "query_id": "1",
        "passages": {
            "is_selected": [1],
            "English_passages": ["The quick brown fox jumps over the lazy dog."],
            "Translated_passages": ["तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर से कूद जाती है।"]
        }
    }
    chunking_latencies = []
    for _ in range(50):
        start = time.time()
        chunker.process_record(record)
        chunking_latencies.append((time.time() - start) * 1000)
        
    # 2. Benchmark Retrieval
    print("Benchmarking Retrieval...")
    retriever = VectorRetriever(db_path=":memory:")
    retrieval_latencies = []
    for _ in range(50):
        start = time.time()
        retriever.retrieve("भारत की राजधानी क्या है?", lang_code="hi")
        retrieval_latencies.append((time.time() - start) * 1000)
        
    # 3. Benchmark API / Orchestrator (Skipping STT for speed testing generation)
    print("Benchmarking Generation...")
    orchestrator = RAGOrchestrator(db_path=":memory:")
    generation_latencies = []
    for _ in range(10):
        start = time.time()
        orchestrator.generate_answer("भारत की राजधानी क्या है?", ["भारत की राजधानी नई दिल्ली है।"])
        generation_latencies.append((time.time() - start) * 1000)

    # Prepare Data
    data = []
    for l in chunking_latencies:
        data.append({"Component": "Chunking", "Latency (ms)": l})
    for l in retrieval_latencies:
        data.append({"Component": "Retrieval", "Latency (ms)": l})
    for l in generation_latencies:
        data.append({"Component": "LLM Generation", "Latency (ms)": l})
        
    df = pd.DataFrame(data)
    
    # Plotting
    os.makedirs("test/test_result", exist_ok=True)
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
    
    # 1. Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Component", y="Latency (ms)", data=df)
    plt.title("Pipeline Component Latencies")
    plt.yscale("log")  # Log scale since generation is much slower than chunking/retrieval
    plt.ylabel("Latency (ms) [Log Scale]")
    plt.tight_layout()
    plt.savefig("test/test_result/component_latency_boxplot.png", dpi=300)
    plt.close()
    
    # 2. P50/P70/P100 Bar Plot
    stats = []
    for comp in ["Chunking", "Retrieval", "LLM Generation"]:
        comp_data = df[df["Component"] == comp]["Latency (ms)"]
        stats.append({
            "Component": comp,
            "P50": np.percentile(comp_data, 50),
            "P70": np.percentile(comp_data, 70),
            "P100": np.max(comp_data)
        })
        
    df_stats = pd.DataFrame(stats).melt(id_vars="Component", var_name="Metric", value_name="Latency (ms)")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Component", y="Latency (ms)", hue="Metric", data=df_stats)
    plt.title("P50 / P70 / P100 Latency Metrics")
    plt.yscale("log")
    plt.ylabel("Latency (ms) [Log Scale]")
    plt.tight_layout()
    plt.savefig("test/test_result/latency_metrics_bar.png", dpi=300)
    plt.close()
    
    print("\nBenchmarks complete! Plots saved to test/test_result/")
    print(pd.DataFrame(stats))

if __name__ == "__main__":
    run_benchmarks()
