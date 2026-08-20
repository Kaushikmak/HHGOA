import os
import time
from typing import Dict, Any, List
from google import genai
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=env_path)

from rag.retriever import VectorRetriever
from audio.audiotranscriber import transcribe_audio
from harness.guardrails import check_off_topic, check_groundedness

class RAGOrchestrator:
    def __init__(self, db_path: str = "qdrant_db"):
        self.retriever = VectorRetriever(db_path=db_path)
        
    def generate_answer(self, query: str, context: List[str]) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not set in environment."
            
        client = genai.Client(api_key=api_key)
        
        context_str = "\n".join(context)
        # Merged Prompt: Handles Answer Generation + Guardrails) in ONE network hop
        prompt = f"""
        You are a strict, helpful RAG assistant. 
        1. Guardrail: If the question is overtly harmful or completely off-topic, output EXACTLY 'REJECTED_OFF_TOPIC'.
        2. Groundedness: If the context does NOT contain the answer, output EXACTLY 'NOT_FOUND'.
        3. Otherwise, answer the question STRICTLY using the provided context. Do not make up facts.
        Answer in the same language as the user's question. Be concise.
        
        Context:
        {context_str}
        
        Question: {query}
        """
        
        try:
            # Using gemini-1.5-flash-8b for the fastest time-to-first-token
            response = client.models.generate_content(
                model='gemini-1.5-flash-8b',
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    max_output_tokens=150,
                    temperature=0.0,
                )
            )
            ans = response.text.strip()
            
            if "REJECTED_OFF_TOPIC" in ans:
                return "I'm sorry, I cannot answer that request as it appears to be off-topic or unsafe."
            if "NOT_FOUND" in ans:
                return "I apologize, but I could not find a verified answer based solely on the provided documents."
                
            return ans
        except Exception as e:
            return f"Error during generation: {e}"

    def process_voice_query(self, audio_file_path: str, lang_code: str = "hi") -> Dict[str, Any]:
        """
        End-to-end pipeline: Voice -> STT -> Retrieval -> Generation (with integrated guardrails)
        Returns structured output with latency metrics.
        """
        metrics = {}
        
        # Transcription
        start_t = time.time()
        transcript = transcribe_audio(audio_file_path)
        metrics["transcription_latency_ms"] = (time.time() - start_t) * 1000
        
        if not transcript:
            return {"error": "Failed to transcribe audio", "metrics": metrics}
            
        # Retrieval
        start_t = time.time()
        context = self.retriever.retrieve(transcript, lang_code=lang_code)
        metrics["retrieval_latency_ms"] = (time.time() - start_t) * 1000
        
        # Generation with Guardrails
        start_t = time.time()
        answer = self.generate_answer(transcript, context)
        metrics["generation_latency_ms"] = (time.time() - start_t) * 1000
        
        metrics["total_pipeline_latency_ms"] = sum(metrics.values())
        
        return {
            "transcript": transcript,
            "context": context,
            "answer": answer,
            "metrics": metrics
        }

