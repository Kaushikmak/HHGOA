import os
import time
from typing import Dict, Any, List
from google import genai
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env')
load_dotenv(dotenv_path=env_path)

from rag.retriever import VectorRetriever
from audio.audiotranscriber import transcribe_audio
from harness.guardrails import (
    check_input_safety, 
    verify_nli_entailment, 
    build_grounded_prompt, 
    REFUSAL_MESSAGES
)

class RAGOrchestrator:
    def __init__(self, db_path: str = "qdrant_db"):
        self.retriever = VectorRetriever(db_path=db_path)
        
    def generate_answer(self, query: str, context: List[str], lang_code: str = "en") -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Error: GEMINI_API_KEY not set in environment."
            
        client = genai.Client(api_key=api_key)
        
        # Merged Prompt: Handles Answer Generation + Guardrails
        prompt = build_grounded_prompt(query, context, lang_code)
        
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
            return ans
        except Exception as e:
            return f"INSUFFICIENT_DATA (Error: {e})"

    def process_voice_query(self, audio_file_path: str, lang_code: str = "hi") -> Dict[str, Any]:
        """
        End-to-end pipeline: Voice -> STT -> Retrieval -> Generation (with integrated ML guardrails)
        Returns structured output with latency metrics.
        """
        metrics = {}
        
        # Transcription
        start_t = time.time()
        transcript = transcribe_audio(audio_file_path)
        metrics["transcription_latency_ms"] = (time.time() - start_t) * 1000
        
        if not transcript:
            return {"error": "Failed to transcribe audio", "metrics": metrics, "status": "ERROR"}
            
        # GUARD 1: Input Safety (Pre-Retrieval)
        start_t = time.time()
        is_safe = check_input_safety(transcript)
        metrics["input_safety_latency_ms"] = (time.time() - start_t) * 1000
        
        if not is_safe:
            metrics["total_pipeline_latency_ms"] = sum(metrics.values())
            return {
                "transcript": transcript,
                "context": [],
                "answer": "Your request could not be processed due to safety guidelines.",
                "metrics": metrics,
                "status": "REJECTED_UNSAFE_INPUT"
            }
            
        # Retrieval
        start_t = time.time()
        context = self.retriever.retrieve(transcript, lang_code=lang_code)
        metrics["retrieval_latency_ms"] = (time.time() - start_t) * 1000
        
        # GUARD 2: Empty Context Bypassing
        if not context:
            localized_refusal = REFUSAL_MESSAGES.get(lang_code, REFUSAL_MESSAGES["en"])
            metrics["total_pipeline_latency_ms"] = sum(metrics.values())
            return {
                "transcript": transcript,
                "context": context,
                "answer": localized_refusal,
                "metrics": metrics,
                "status": "ABSTAINED_NO_GROUNDING"
            }
        
        # Generation
        start_t = time.time()
        raw_answer = self.generate_answer(transcript, context, lang_code)
        metrics["generation_latency_ms"] = (time.time() - start_t) * 1000
        
        # GUARD 3: Output Hallucination Prevention
        start_t = time.time()
        is_entailed = verify_nli_entailment(transcript, context, raw_answer)
        metrics["nli_verification_latency_ms"] = (time.time() - start_t) * 1000
        
        localized_refusal = REFUSAL_MESSAGES.get(lang_code, REFUSAL_MESSAGES["en"])
        if not is_entailed or "INSUFFICIENT_DATA" in raw_answer:
            final_answer = localized_refusal
        else:
            final_answer = raw_answer
        
        metrics["total_pipeline_latency_ms"] = sum(metrics.values())
        
        return {
            "transcript": transcript,
            "context": context,
            "answer": final_answer,
            "metrics": metrics,
            "status": "SUCCESS"
        }
