import os

# The rate limiter binds its storage backend at import time — app/utils/limiter.py
# reads settings.redis_url — and Settings loads the developer's .env, which points
# REDIS_URL at a real Upstash instance. Importing app.app below would therefore make
# every rate-limited route reach out over the network, failing the suite on DNS
# rather than on anything the tests assert. Environment variables outrank .env in
# pydantic-settings, so pinning it here (before any app.* import) is enough. CI never
# hit this because it has no .env file.
os.environ["REDIS_URL"] = "memory://"

# Same failure mode, different setting: app/database/db.py builds its engine at
# import time from settings.database_url, so the `client` fixture's Settings
# override never reaches it. Without this line the suite reads *and writes* the
# developer's real local database, and tests silently depend on whatever rows
# happen to be sitting in it.
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from app.app import app  # noqa: E402
from app.config import Settings, get_settings  # noqa: E402
from app.database.db import Base, SessionLocal, engine  # noqa: E402
from app.database.models import APIKey, UploadedFile  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def mock_observability():
    """Prevent CloudTraceSpanExporter from blocking on GCP network during tests."""
    with patch("app.utils.observability.CloudTraceSpanExporter", MagicMock()):
        yield


# Attachment that tests reference by a fixed ID. It used to be an ambient row in
# the developer's local database; seeding it here makes the suite self-contained.
SEEDED_FILE_ID = "00000000-0000-0000-0000-000000000001"

# Two real, non-master API keys. Every ownership filter is written
# `if api_key.id != "master"`, so a suite that only ever authenticates as master
# exercises the bypass and never the filter -- which is how the 2026-08-30 cache
# leak stayed invisible. T6's regression test needs two distinct owners.
SEEDED_OWNER_ID = "00000000-0000-0000-0000-0000000000aa"
SEEDED_OTHER_OWNER_ID = "00000000-0000-0000-0000-0000000000bb"


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    # Initialize the database tables once for the test session
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        if db.get(APIKey, "master") is None:
            db.add(APIKey(id="master", name="Master Key", hashed_key="seeded-master"))
        for owner_id, label in (
            (SEEDED_OWNER_ID, "Delegated Owner"),
            (SEEDED_OTHER_OWNER_ID, "Other Owner"),
        ):
            if db.get(APIKey, owner_id) is None:
                db.add(APIKey(id=owner_id, name=label, hashed_key=f"seeded-{owner_id}"))
        if db.get(UploadedFile, SEEDED_FILE_ID) is None:
            db.add(
                UploadedFile(
                    id=SEEDED_FILE_ID,
                    gemini_file_name="files/seeded",
                    gemini_file_uri="https://generativelanguage.googleapis.com/v1beta/files/seeded",
                    mime_type="application/pdf",
                    size_bytes=1024,
                    display_name="seeded.pdf",
                    owner_id="master",
                )
            )
        db.commit()
    finally:
        db.close()

    yield


@pytest.fixture(scope="session", autouse=True)
def mock_gemini_client_global():
    """Globally mock the Gemini client and LangChain LLM to prevent real API calls."""
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
        patch("app.services.gemini.global_client", mock_client),
        # The regional/global fallback call path lives in one module now (F3).
        patch("app.services.genai_calls.client", mock_client),
        patch("app.services.genai_calls.global_client", mock_client),
        patch("app.services.caches.client", mock_client),
        patch("app.services.health.client", mock_client),
        patch("app.services.gemini.build_llm", return_value=mock_llm_instance),
        # Direct .invoke() sites use the regional/global fallback wrapper (F9).
        patch(
            "app.services.gemini.build_llm_with_region_fallback",
            return_value=mock_llm_instance,
        ),
        patch("app.agents.base.build_llm", return_value=mock_llm_instance),
        patch("app.services.agents.run_once", side_effect=_mock_run_once),
        patch("app.services.agents.get_checkpointer", return_value=None),
    ):
        yield mock_client


@pytest.fixture(scope="session", autouse=True)
def mock_health_pinecone():
    """`/api/v1/health` probes Pinecone directly — keep the suite off the network."""
    with patch("app.services.health.Pinecone", MagicMock()):
        yield


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


@pytest.fixture(scope="session")
def delegated_owner_id():
    """A real, non-master owner for delegated (x-internal-key) invocations."""
    return SEEDED_OWNER_ID


@pytest.fixture(scope="session")
def other_owner_id():
    """A second real owner, for asserting one owner cannot reach another's resources."""
    return SEEDED_OTHER_OWNER_ID


@pytest.fixture()
def gemini_caches(mock_gemini_client_global):
    """A MagicMock caches service attached to the global Gemini client.

    Each test gets a fresh mock.  Replaces the repeated
    ``mock_caches = MagicMock(); mock_gemini_client_global.caches = mock_caches``
    pattern that appeared in every cache-related test file.
    """
    mock_caches = MagicMock()
    mock_gemini_client_global.caches = mock_caches
    return mock_caches


@pytest.fixture()
def make_gemini_cache():
    """Factory fixture: returns a MagicMock with the six attributes every Gemini cache mock needs.

    Usage::

        cache = make_gemini_cache(name="cachedContents/abc123", display_name="my-cache")
    """

    def _make(
        name: str = "cachedContents/test-cache",
        model: str = "gemini-2.5-flash",
        display_name: str = "test-cache",
        ttl=None,
        create_time=None,
        expire_time=None,
    ):
        cache = MagicMock()
        cache.name = name
        cache.model = model
        cache.display_name = display_name
        cache.ttl = ttl
        cache.create_time = create_time
        cache.expire_time = expire_time
        return cache

    return _make
