import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.app import app
from app.config import Settings, get_settings
from app.database.db import Base, engine


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Initialize the database tables once for the test session
    Base.metadata.create_all(bind=engine)
    yield
    # Optional: Drop tables after session if needed
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def mock_gemini_client_global():
    """Globally mock the Gemini client where it's used."""
    mock_client = MagicMock()
    
    # Mock the structure of the client
    mock_client.models.list_models.return_value = []
    
    # Mock generate_content response
    mock_response = MagicMock()
    mock_response.text = '{"foo": "bar"}'
    mock_client.models.generate_content.return_value = mock_response
    
    # Use context manager to patch the module-level client
    with patch("app.services.gemini.client", mock_client):
        yield mock_client

@pytest.fixture
def client():
    # Define test settings with dummy values
    test_settings = Settings(
        database_url="sqlite:///./test.db",
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
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up overrides
    app.dependency_overrides.clear()

@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-master-key"}
