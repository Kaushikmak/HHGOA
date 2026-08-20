import os
import pytest
from unittest.mock import patch

# Mock db path to :memory: so we don't hit the file lock from the running server
with patch("qdrant_client.QdrantClient.__init__", return_value=None):
    from fastapi.testclient import TestClient
    from rag.chunking import IndicRAGChunker
    from harness.orchestrator import RAGOrchestrator
    from api.main import app

def test_chunking_logic():
    chunker = IndicRAGChunker(lang_code="hi", chunk_size_limit=200)
    record = {
        "query_id": "123",
        "passages": {
            "is_selected": [1, 0],
            "English_passages": ["This is a test passage.", "Another test."],
            "Translated_passages": ["यह एक परीक्षण है।", "एक और परीक्षण।"]
        }
    }
    nodes = chunker.process_record(record)
    assert len(nodes) > 0
    assert nodes[0].metadata["query_id"] == "123"

def test_api_root():
    # We patch orchestrator so it doesn't fail on qdrant lock
    with patch("api.routes.orchestrator"):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Voice-Enabled Indic RAG API"}
