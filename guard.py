import asyncio
import time
import re
import os
import numpy as np
from typing import Dict, Any, List, Optional
from qdrant_client import AsyncQdrantClient, models
from google import genai
import warnings

# Suppress huggingface warnings for clean logs
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# ML Model Initialization (Lazy Loading)
# ---------------------------------------------------------------------------
class GuardModels:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GuardModels, cls).__new__(cls)
            cls._instance.topic_encoder = None
            cls._instance.safety_classifier = None
            cls._instance.nli_encoder = None
            
            # Predefined allowed topics for similarity checking
            cls._instance.allowed_topics = [
                "agriculture and farming techniques",
                "weather, climate and crop forecasting",
                "government schemes, subsidies and loans",
                "market prices for crops and vegetables",
                "pest control, fertilizers and soil health",
                "general greetings, assistance and support"
            ]
            cls._instance.allowed_topic_embeddings = None
        return cls._instance

    def get_safety_classifier(self):
        """Loads a lightweight multilingual text classification pipeline."""
        if self.safety_classifier is None:
            from transformers import pipeline
            # Fallback to a fast classifier; in production replace with custom quantized ONNX mBERT
            print("Loading ML Safety Classifier...")
            self.safety_classifier = pipeline("text-classification", model="michellejieli/inappropriate_text_classifier", device="cpu")
        return self.safety_classifier

    def get_topic_encoder(self):
        """Loads a fast multilingual sentence transformer for off-topic catching."""
        if self.topic_encoder is None:
            from sentence_transformers import SentenceTransformer
            print("Loading ML Topic Encoder...")
            # Fast multilingual model
            self.topic_encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")
            self.allowed_topic_embeddings = self.topic_encoder.encode(self.allowed_topics)
        return self.topic_encoder
        
    def get_nli_encoder(self):
        """Loads a high-speed cross-encoder for Fast NLI (Output hallucination catch)."""
        if self.nli_encoder is None:
            from sentence_transformers import CrossEncoder
            print("Loading ML NLI Encoder...")
            self.nli_encoder = CrossEncoder('cross-encoder/nli-deberta-v3-xsmall', device="cpu")
        return self.nli_encoder

# ---------------------------------------------------------------------------
# 1. Input Guardrail: Fast Keyword + ML Heuristics (Pre-Retrieval)
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS = {
    "system prompt", "ignore previous instructions", "bypass", "jailbreak",
}
PATTERN_REGEX = re.compile(r'\b(' + '|'.join(map(re.escape, BLOCKED_PATTERNS)) + r')\b', re.IGNORECASE)

def _run_ml_input_safety(query: str) -> bool:
    """
    Synchronous ML inference block. 
    Returns True if SAFE, False if UNSAFE or OFF-TOPIC.
    """
    # 1. Fast Regex Check (Sub-1ms)
    if PATTERN_REGEX.search(query):
        return False
        
    try:
        models_manager = GuardModels()
        
        # 2. Multilingual Safety Filter (Binary Classifier)
        # Assuming model outputs 'inappropriate' or 'appropriate' or similar
        classifier = models_manager.get_safety_classifier()
        safety_result = classifier(query[:512])[0] # Truncate to avoid context limit
        # If the model explicitly flags it, block it. (Adjust label matching based on model)
        if safety_result['label'] == 'INAPPROPRIATE' or safety_result['label'] == 'LABEL_1':
            return False

        # 3. Off-Topic Catching (Semantic Similarity)
        encoder = models_manager.get_topic_encoder()
        query_emb = encoder.encode(query)
        allowed_embs = models_manager.allowed_topic_embeddings
        
        # Cosine similarity against allowed topics
        from numpy import dot
        from numpy.linalg import norm
        similarities = dot(allowed_embs, query_emb) / (norm(allowed_embs, axis=1) * norm(query_emb))
        
        max_sim = np.max(similarities)
        
        # If similarity to nearest allowed topic is too low, reject
        if max_sim < 0.20:  # Threshold tuned for typical out-of-domain rejection
            return False
            
        return True
    except Exception as e:
        print(f"ML Safety Check Error: {e}")
        # Fail-open if ML models fail to load/infer, or you can fail-closed depending on policy
        return True

async def check_input_safety(query: str) -> bool:
    """
    Async wrapper for ML input safety checking to prevent blocking the event loop.
    Executes the thread-blocking ML inference in a separate thread.
    """
    return await asyncio.to_thread(_run_ml_input_safety, query)


# ---------------------------------------------------------------------------
# 2. Retrieval Guardrail: Qdrant Language Isolation & Score Thresholding
# ---------------------------------------------------------------------------
async def retrieve_grounded_context(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    target_lang: str,
    score_threshold: float = 0.65,
    limit: int = 3
) -> List[Dict[str, Any]]:
    lang_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="target_lang",
                match=models.MatchValue(value=target_lang)
            )
        ]
    )

    search_result = await client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=lang_filter,
        score_threshold=score_threshold,
        limit=limit
    )

    return [hit.payload for hit in search_result if hit.payload is not None]


# ---------------------------------------------------------------------------
# 3. Output Guardrail: Localized Safe Refusals, Strict Prompting & Fast NLI
# ---------------------------------------------------------------------------
REFUSAL_MESSAGES = {
    "as": "প্ৰদান কৰা প্ৰসংগত মই এই প্ৰশ্নটোৰ উত্তৰ বিচাৰি নাপালোঁ।",
    "bn": "আমি প্রদত্ত তথ্যে এই প্রশ্নের উত্তর খুঁজে পাইনি।",
    "gu": "પૂરા પાડવામાં આવેલ સંદર્ભમાં મને આ પ્રશ્નનો જવાબ મળ્યો નથી.",
    "hi": "मुझे प्रदान किए गए संदर्भ में इस प्रश्न का उत्तर नहीं मिला।",
    "kn": "ಒದಗಿಸಿದ ಸಂದರ್ಭದಲ್ಲಿ ಈ ಪ್ರಶ್ನೆಗೆ ನನಗೆ ಉತ್ತರ ಸಿಗಲಿಲ್ಲ.",
    "ml": "നൽകിയ വിവരങ്ങളിൽ ഈ ചോദ്യത്തിനുള്ള ഉത്തരം കണ്ടെത്താനായില്ല.",
    "mr": "मला दिलेल्या संदर्भात या प्रश्नाचे उत्तर सापडले नाही.",
    "ne": "मैले प्रदान गरिएको सन्दर्भमा यस प्रश्नको उत्तर फेला पार्न सकिन।",
    "or": "ପ୍ରଦତ୍ତ ପ୍ରସଙ୍ଗରେ ମୁଁ ଏହି ପ୍ରଶ୍ନର ଉତ୍ତର ପାଇଲି ନାହିଁ।",
    "pa": "ਮੈਨੂੰ ਪ੍ਰਦਾਨ ਕੀਤੇ ਗਏ ਸੰਦਰਭ ਵਿੱਚ ਇਸ ਸਵਾਲ ਦਾ ਜਵਾਬ ਨਹੀਂ ਮਿਲਿਆ।",
    "sa": "प्रदत्ते सन्दर्भे मया अस्य प्रश्नस्य उत्तरं 단 प्राप्तम्।",
    "ta": "வழங்கப்பட்ட சூழலில் இந்தக் கேள்விக்கான பதில் கிடைக்கவில்லை.",
    "te": "అందించిన సమాచారంలో ఈ ప్రశ్నకు సమాధానం దొరకలేదు.",
    "ur": "مجھے فراہم کردہ سیاق و سباق میں اس سوال کا جواب نہیں ملا۔",
    "en": "I could not find the answer in the provided context."
}

def build_grounded_prompt(query: str, contexts: List[Dict[str, Any]], lang: str) -> str:
    context_text = "\n\n".join([c.get("text", "") for c in contexts])
    return f"""You are a strictly grounded factual assistant.
Answer the question ONLY using the context provided below in the language code '{lang}'.
If the context does not contain sufficient facts to answer, reply EXACTLY with: 'INSUFFICIENT_DATA'.

Context:
{context_text}

Question:
{query}

Answer:"""

def _run_ml_output_nli(query: str, contexts: List[Dict[str, Any]], llm_answer: str) -> bool:
    """
    Fast NLI Cross-Encoder check to verify logical entailment.
    Returns True if grounded, False if hallucinated/contradiction.
    """
    try:
        models_manager = GuardModels()
        nli_encoder = models_manager.get_nli_encoder()
        
        context_text = " ".join([c.get("text", "") for c in contexts])
        
        # Create (premise, hypothesis) pair
        # Premise: Context, Hypothesis: LLM Answer
        scores = nli_encoder.predict([(context_text, llm_answer)])
        
        # For NLI models, typically: 0 = contradiction, 1 = entailment, 2 = neutral
        # We want to ensure it is not a contradiction.
        # Deberta-v3 NLI specific label mapping varies, but generally argmax gives the class.
        predicted_class = np.argmax(scores, axis=1)[0]
        
        # Assuming contradiction is class 0 or high contradiction score
        if predicted_class == 0:  
            return False
            
        return True
    except Exception as e:
        print(f"ML NLI Check Error: {e}")
        return True

async def verify_nli_entailment(query: str, contexts: List[Dict[str, Any]], llm_answer: str) -> bool:
    if "INSUFFICIENT_DATA" in llm_answer:
        return True
    return await asyncio.to_thread(_run_ml_output_nli, query, contexts, llm_answer)

# ---------------------------------------------------------------------------
# 4. Orchestration Harness (Parallel Safety + Retrieval)
# ---------------------------------------------------------------------------
async def voice_rag_guardrail_harness(
    query_text: str,
    query_vector: List[float],
    detected_lang: str,
    qdrant_client: AsyncQdrantClient,
    collection_name: str
) -> Dict[str, Any]:
    
    t_start = time.perf_counter()
    timings = {}

    # STEP 1: Execute ML Input Safety and Retrieval Concurrently
    # We use asyncio.wait to cancel retrieval if safety fails early
    t0 = time.perf_counter()
    
    safety_task = asyncio.create_task(check_input_safety(query_text))
    retrieval_task = asyncio.create_task(
        retrieve_grounded_context(
            client=qdrant_client,
            collection_name=collection_name,
            query_vector=query_vector,
            target_lang=detected_lang,
            score_threshold=0.65
        )
    )

    done, pending = await asyncio.wait(
        [safety_task, retrieval_task], 
        return_when=asyncio.FIRST_COMPLETED
    )

    is_safe = True
    if safety_task in done:
        is_safe = safety_task.result()
        if not is_safe:
            retrieval_task.cancel()  # Abort DB retrieval immediately
            timings["input_safety_ms"] = (time.perf_counter() - t0) * 1000
            return {
                "status": "REJECTED_UNSAFE_INPUT",
                "response": "Your request could not be processed due to safety guidelines.",
                "timings": timings,
                "total_ms": (time.perf_counter() - t_start) * 1000
            }

    # Wait for retrieval if not finished
    if retrieval_task not in done:
        await retrieval_task

    # Catch edge case where retrieval finished first but safety failed
    is_safe = safety_task.result()
    if not is_safe:
        timings["input_safety_ms"] = (time.perf_counter() - t0) * 1000
        return {
            "status": "REJECTED_UNSAFE_INPUT",
            "response": "Your request could not be processed due to safety guidelines.",
            "timings": timings,
            "total_ms": (time.perf_counter() - t_start) * 1000
        }

    retrieved_chunks = retrieval_task.result()
    timings["input_safety_and_retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # GUARD 2: Context Grounding / Empty Context Bypassing
    if not retrieved_chunks:
        localized_refusal = REFUSAL_MESSAGES.get(detected_lang, REFUSAL_MESSAGES["en"])
        return {
            "status": "ABSTAINED_NO_GROUNDING",
            "response": localized_refusal,
            "timings": timings,
            "total_ms": (time.perf_counter() - t_start) * 1000
        }

    # STEP 2: Grounded Prompt Formulation
    prompt = build_grounded_prompt(query_text, retrieved_chunks, detected_lang)

    # STEP 3: LLM Generation
    t_llm = time.perf_counter()
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            raw_llm_response = response.text.strip()
        except Exception as e:
            raw_llm_response = f"INSUFFICIENT_DATA (Error: {str(e)})"
    else:
        raw_llm_response = "Generated answer (Please set GEMINI_API_KEY)"
        
    timings["llm_generation_ms"] = (time.perf_counter() - t_llm) * 1000

    # GUARD 3: Fast NLI Hallucination Verification
    t_nli = time.perf_counter()
    is_entailed = await verify_nli_entailment(query_text, retrieved_chunks, raw_llm_response)
    timings["output_nli_ms"] = (time.perf_counter() - t_nli) * 1000
    
    if not is_entailed or "INSUFFICIENT_DATA" in raw_llm_response:
        final_answer = REFUSAL_MESSAGES.get(detected_lang, REFUSAL_MESSAGES["en"])
    else:
        final_answer = raw_llm_response

    timings["total_ms"] = (time.perf_counter() - t_start) * 1000

    return {
        "status": "SUCCESS",
        "response": final_answer,
        "contexts_used": len(retrieved_chunks),
        "timings": timings
    }