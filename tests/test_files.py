from unittest.mock import MagicMock
from fastapi.testclient import TestClient


def test_files_workflow_happy_path(client: TestClient, auth_headers, mock_gemini_client_global):
    # 1. Mock the files.upload call
    mock_file = MagicMock()
    mock_file.name = "files/test-file-123"
    mock_file.uri = "https://generativelanguage.googleapis.com/v1beta/files/test-file-123"
    mock_gemini_client_global.files.upload.return_value = mock_file

    # 2. Upload file
    files = {"file": ("test.txt", b"Hello Antigravity!", "text/plain")}
    response = client.post(
        "/api/v1/files/upload",
        files=files,
        headers=auth_headers
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["display_name"] == "test.txt"
    assert data["mime_type"] == "text/plain"
    assert data["gemini_file_name"] == "files/test-file-123"
    assert data["gemini_file_uri"] == "https://generativelanguage.googleapis.com/v1beta/files/test-file-123"
    file_id = data["id"]

    # 3. List files
    response = client.get(
        "/api/v1/files/",
        headers=auth_headers
    )
    assert response.status_code == 200
    files_list = response.json()["data"]
    assert len(files_list) > 0
    assert any(f["id"] == file_id for f in files_list)

    # 4. Mock files.delete call
    mock_gemini_client_global.files.delete.return_value = None

    # 5. Delete file
    response = client.delete(
        f"/api/v1/files/{file_id}",
        headers=auth_headers
    )
    assert response.status_code == 200
    assert "Successfully deleted" in response.json()["data"]["message"]

    # 6. Verify it's gone from list
    response = client.get(
        "/api/v1/files/",
        headers=auth_headers
    )
    assert response.status_code == 200
    files_list = response.json()["data"]
    assert not any(f["id"] == file_id for f in files_list)


def test_upload_without_auth_fails(client: TestClient):
    files = {"file": ("test.txt", b"Hello Antigravity!", "text/plain")}
    response = client.post(
        "/api/v1/files/upload",
        files=files
    )
    assert response.status_code in [401, 422]
