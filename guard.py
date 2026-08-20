import asyncio
import time
from typing import Dict, Any, List, Optional
from qdrant_client import AsyncQdrantClient, models

# ---------------------------------------------------------------------------
# 1. Input Guardrail: Fast Keyword / Semantic Heuristics
# ---------------------------------------------------------------------------
BLOCKED_PATTERNS = {
    "system prompt", "ignore previous instructions", "bypass", "jailbreak",
    # Add common jailbreak terms across supported Indic scripts if needed
}

async def check_input_safety(query: str) -> bool:
    """
    Sub-5ms rule-based check. For deeper checks, use a lightweight 
    ONNX-quantized multilingual classification model.
    """
    query_lower = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return False
    return True


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
    """
    Retrieves context with strict language isolation and similarity cutoffs.
    """
    # Strict Metadata Filtering: Ensure Hindi queries only match Hindi documents
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
        score_threshold=score_threshold,  # Rejects weak matches automatically
        limit=limit
    )

    return [hit.payload for hit in search_result if hit.payload is not None]


# ---------------------------------------------------------------------------
# 3. Output Guardrail: Localized Safe Refusals & Strict Prompting
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
    "sa": "प्रदत्ते सन्दर्भे मया अस्य प्रश्नस्य उत्तरं न प्राप्तम्।",
    "ta": "வழங்கப்பட்ட சூழலில் இந்தக் கேள்விக்கான பதில் கிடைக்கவில்லை.",
    "te": "అందించిన సమాచారంలో ఈ ప్రశ్నకు సమాధానం దొరకలేదు.",
    "ur": "مجھے فراہم کردہ سیاق و سباق میں اس سوال کا جواب نہیں ملا۔",
    "en": "I could not find the answer in the provided context."
}

def build_grounded_prompt(query: str, contexts: List[Dict[str, Any]], lang: str) -> str:
    context_text = "\n\n".join([c.get("text", "") for c in contexts])
    return f"""You are a strictly grounded factual assistant.
Answer the question ONLY using the context provided below in the language code '{lang}'.
If the context does not contain sufficient facts to answer, reply EXACTLY with: 'INSUFFICIENT_CONTEXT'.

Context:
{context_text}

Question:
{query}

Answer:"""


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

    # STEP 1: Execute Input Safety and Retrieval Concurrently
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

    is_safe, retrieved_chunks = await asyncio.gather(safety_task, retrieval_task)
    timings["input_safety_and_retrieval_ms"] = (time.perf_counter() - t0) * 1000

    # GUARD 1: Input Violation Check
    if not is_safe:
        return {
            "status": "REJECTED_UNSAFE_INPUT",
            "response": "Your request could not be processed due to safety guidelines.",
            "timings": timings,
            "total_ms": (time.perf_counter() - t_start) * 1000
        }

    # GUARD 2: Context Grounding / Abstention Check
    if not retrieved_chunks:
        # Fast exit: Bypasses LLM generation entirely (<15ms total latency)
        localized_refusal = REFUSAL_MESSAGES.get(detected_lang, REFUSAL_MESSAGES["en"])
        return {
            "status": "ABSTAINED_NO_GROUNDING",
            "response": localized_refusal,
            "timings": timings,
            "total_ms": (time.perf_counter() - t_start) * 1000
        }

    # STEP 2: Grounded Prompt Formulation
    prompt = build_grounded_prompt(query_text, retrieved_chunks, detected_lang)

    # STEP 3: LLM Generation (Fast Engine / Streaming)
    t_llm = time.perf_counter()
    # Replace with your actual LLM inference call
    raw_llm_response = "Generated answer..." 
    timings["llm_generation_ms"] = (time.perf_counter() - t_llm) * 1000

    # GUARD 3: Output Hallucination Catch
    if "INSUFFICIENT_CONTEXT" in raw_llm_response:
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