import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from harness.orchestrator import RAGOrchestrator

router = APIRouter()
_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator

@router.post("/ask")
async def ask_question(audio: UploadFile = File(...), lang_code: str = Form("hi")):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")
        
    temp_dir = "/tmp/rag_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, audio.filename)
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
            
        orchestrator = get_orchestrator()
        result = orchestrator.process_voice_query(temp_path, lang_code=lang_code)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
