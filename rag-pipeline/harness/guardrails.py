import re
import os
import numpy as np
from typing import Dict, Any, List
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

def check_input_safety(query: str) -> bool:
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
        classifier = models_manager.get_safety_classifier()
        safety_result = classifier(query[:512])[0]
        if safety_result['label'] == 'INAPPROPRIATE' or safety_result['label'] == 'LABEL_1':
            return False

        # 3. Off-Topic Catching (Semantic Similarity)
        encoder = models_manager.get_topic_encoder()
        query_emb = encoder.encode(query)
        allowed_embs = models_manager.allowed_topic_embeddings
        
        from numpy import dot
        from numpy.linalg import norm
        similarities = dot(allowed_embs, query_emb) / (norm(allowed_embs, axis=1) * norm(query_emb))
        
        max_sim = np.max(similarities)
        if max_sim < 0.20:
            return False
            
        return True
    except Exception as e:
        print(f"ML Safety Check Error: {e}")
        return True


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

def build_grounded_prompt(query: str, contexts: List[str], lang: str) -> str:
    context_text = "\n\n".join(contexts)
    return f"""You are a strictly grounded factual assistant.
Answer the question ONLY using the context provided below in the language code '{lang}'.
If the context does not contain sufficient facts to answer, reply EXACTLY with: 'INSUFFICIENT_DATA'.

Context:
{context_text}

Question:
{query}

Answer:"""

def verify_nli_entailment(query: str, contexts: List[str], llm_answer: str) -> bool:
    """
    Fast NLI Cross-Encoder check to verify logical entailment.
    Returns True if grounded, False if hallucinated/contradiction.
    """
    if "INSUFFICIENT_DATA" in llm_answer:
        return True
        
    try:
        models_manager = GuardModels()
        nli_encoder = models_manager.get_nli_encoder()
        
        context_text = " ".join(contexts)
        
        scores = nli_encoder.predict([(context_text, llm_answer)])
        predicted_class = np.argmax(scores, axis=1)[0]
        
        if predicted_class == 0:  
            return False
            
        return True
    except Exception as e:
        print(f"ML NLI Check Error: {e}")
        return True
