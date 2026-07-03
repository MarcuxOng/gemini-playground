import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.app import app
from app.config import Settings, get_settings
from app.database.db import Base, engine


@pytest.fixture(scope="session", autouse=True)
def mock_observability():
    """Prevent CloudTraceSpanExporter from blocking on GCP network during tests."""
    with patch("app.utils.observability.CloudTraceSpanExporter", MagicMock()):
        yield


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Initialize the database tables once for the test session
    Base.metadata.create_all(bind=engine)
    yield
    # Optional: Drop tables after session if needed
    # Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session", autouse=True)
def mock_gemini_client_global():
    """Globally mock the Gemini client and LangChain LLM to prevent real API calls."""
    from langchain_core.messages import AIMessage

    mock_client = MagicMock()
    mock_client.models.list_models.return_value = []
    mock_response = MagicMock()
    mock_response.text = '{"foo": "bar"}'
    mock_response.candidates = []
    mock_response.prompt_feedback = None
    mock_client.models.generate_content.return_value = mock_response

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="mocked LLM response")
    mock_llm_instance.ainvoke.return_value = AIMessage(content="mocked LLM response")

    def _mock_run_once(agent, question, lg_config=None):
        return ("mocked LLM response", {"input_tokens": 10, "output_tokens": 20})

    with (
        patch("app.services.gemini.client", mock_client),
        patch("app.services.image.client", mock_client),
        patch("app.services.caches.client", mock_client),
        patch("app.services.gemini.build_llm", return_value=mock_llm_instance),
        patch("app.agents.base.build_llm", return_value=mock_llm_instance),
        patch("app.services.agents.run_once", side_effect=_mock_run_once),
        patch("app.services.agents.get_checkpointer", return_value=None),
    ):
        yield mock_client


@pytest.fixture(scope="session")
def client():
    # Define test settings with dummy values
    test_settings = Settings(
        database_url="sqlite:///./test.db",
        master_api_key="test-master-key",
        internal_api_key="test-internal-key",
        gemini_api_key="test-key",
        gcp_project_id="test-project",
        pinecone_namespace="test-ns",
        pinecone_index_name="test-idx",
        pinecone_api_key="test-key",
        alpha_vantage_api_key="test",
        openweathermap_api_key="test",
        news_api_key="test",
    )

    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def auth_headers():
    return {"X-API-Key": "test-master-key"}


@pytest.fixture(scope="session")
def internal_auth_headers():
    return {"x-internal-key": "test-internal-key"}
