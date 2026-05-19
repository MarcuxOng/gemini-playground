from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import uuid

def test_create_dataset_success(client: TestClient, auth_headers):
    name = f"Test Dataset {uuid.uuid4()}"
    response = client.post(
        "/api/v1/evals/datasets",
        json={
            "name": name,
            "cases": [{"input": "hi", "expected": "hello"}]
        },
        headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["data"]["name"] == name

def test_start_eval_success(client: TestClient, auth_headers):
    # First create a dataset
    name = f"Eval Run Dataset {uuid.uuid4()}"
    ds_resp = client.post(
        "/api/v1/evals/datasets",
        json={
            "name": name,
            "cases": [{"input": "hi", "expected": "hello"}]
        },
        headers=auth_headers
    )
    ds_id = ds_resp.json()["data"]["id"]

    # Mock run_eval service to avoid real LLM calls
    mock_result = {
        "run_id": "test-run-id",
        "metrics": {"passed": 1, "failed": 0, "total": 1, "results": []}
    }
    
    with patch("app.routers.evals.run_eval", return_value=mock_result):
        response = client.post(
            "/api/v1/evals/run",
            json={
                "dataset_id": ds_id,
                "agent_id_or_preset": "research",
                "model": "gemini-1.5-flash"
            },
            headers=auth_headers
        )
        assert response.status_code == 200
        assert response.json()["data"]["run_id"] == "test-run-id"
