import pytest
from fastapi.testclient import TestClient
from app.app import app
from app.config import Settings, get_settings

@pytest.fixture
def client():
    # Define test settings with dummy values
    test_settings = Settings(
        master_api_key="test-master-key",
        gemini_api_key="test-key",
        gcp_project_id="test-project",
        pinecone_namespace="test-ns",
        pinecone_index_name="test-idx",
        pinecone_api_key="test-key",
        alpha_vantage_api_key="test",
        openweathermap_api_key="test",
        news_api_key="test"
    )
    
    # Override the settings dependency
    app.dependency_overrides[get_settings] = lambda: test_settings
    
    client = TestClient(app)
    yield client
    
    # Clean up overrides
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-master-key"}

@pytest.fixture
def mock_gemini_client(monkeypatch):
    class MockClient:
        def __init__(self, *args, **kwargs):
            pass
        class Models:
            def list_models(self):
                return []
        models = Models()

    monkeypatch.setattr("google.genai.Client", MockClient)
    return MockClient
