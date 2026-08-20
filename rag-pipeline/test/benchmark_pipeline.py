import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict
import argparse

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from harness.orchestrator import RAGOrchestrator

def benchmark_pipeline(queries: List[str], lang_code: str = "hi"):
    orchestrator = RAGOrchestrator()
    
    total_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    
    print(f"Benchmarking pipeline with {len(queries)} queries for lang: {lang_code}...")
    
    for i, query in enumerate(queries):
        print(f"Processing query {i+1}/{len(queries)}")
        
        # We skip STT for this specific benchmark since it's text-based
        # We'll just measure from Retrieval to Output
        start_pipeline = time.time()
        
        # Retrieval
        start_retrieval = time.time()
        context = orchestrator.retriever.retrieve(query, lang_code=lang_code)
        retrieval_latency = (time.time() - start_retrieval) * 1000
        
        # Generation
        start_generation = time.time()
        answer = orchestrator.generate_answer(query, context)
        generation_latency = (time.time() - start_generation) * 1000
        
        total_latency = (time.time() - start_pipeline) * 1000
        
        total_latencies.append(total_latency)
        retrieval_latencies.append(retrieval_latency)
        generation_latencies.append(generation_latency)
        
    print("\n--- Latency Analytics ---")
    print(f"Total Queries: {len(queries)}")
    print(f"P50 Total Latency: {np.percentile(total_latencies, 50):.2f} ms")
    print(f"P70 Total Latency: {np.percentile(total_latencies, 70):.2f} ms")
    print(f"P100 Total Latency: {np.max(total_latencies):.2f} ms")
    
    print(f"\nAverage Retrieval: {np.mean(retrieval_latencies):.2f} ms")
    print(f"Average Generation: {np.mean(generation_latencies):.2f} ms")

if __name__ == "__main__":
    test_queries = [
        "What is the capital of India?",
        "How do I cook rice?",
        "Tell me about the history of the Taj Mahal.",
        "What are the symptoms of a cold?",
        "When did India get independence?"
    ]
    benchmark_pipeline(test_queries)
