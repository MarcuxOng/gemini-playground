from unittest.mock import MagicMock, patch

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


def test_upload_to_gcs_in_production(client: TestClient, auth_headers):
    """When ENV=production, files are uploaded to GCS, not Gemini Files API."""
    with patch.dict("os.environ", {"ENV": "production", "GCS_BUCKET": "test-bucket"}):
        with (
            patch("app.routers.files.validate_upload") as mock_validate,
            patch("app.services.gemini.upload_to_gcs") as mock_gcs,
            patch("app.services.gemini.client.files.upload") as mock_gemini_upload,
        ):
            files = {"file": ("photo.png", b"fake-image-data", "image/png")}
            resp = client.post(
                "/api/v1/files/upload",
                files=files,
                headers=auth_headers,
            )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["gemini_file_uri"].startswith("gs://test-bucket/uploads/")
    assert data["mime_type"] == "image/png"
    assert data["display_name"] == "photo.png"
    mock_gcs.assert_called_once()
    mock_gemini_upload.assert_not_called()


def test_delete_gcs_file(client: TestClient, auth_headers):
    """When file URI is gs://, deletion hits GCS, not Gemini Files API."""
    with patch.dict("os.environ", {"ENV": "production", "GCS_BUCKET": "test-bucket"}):
        with (
            patch("app.routers.files.validate_upload") as mock_validate,
            patch("app.services.gemini.upload_to_gcs") as mock_gcs_upload,
            patch("app.services.gemini.delete_from_gcs") as mock_gcs_delete,
        ):
            # Upload to create a GCS-backed record
            files = {"file": ("doc.pdf", b"pdf-data", "application/pdf")}
            upload_resp = client.post(
                "/api/v1/files/upload",
                files=files,
                headers=auth_headers,
            )
            assert upload_resp.status_code == 200
            file_id = upload_resp.json()["data"]["id"]

            # Delete it
            delete_resp = client.delete(
                f"/api/v1/files/{file_id}",
                headers=auth_headers,
            )

    assert delete_resp.status_code == 200
    mock_gcs_delete.assert_called_once()


def test_upload_file_to_gemini_dev_path():
    """In dev (no ENV), upload_file_to_gemini uses Gemini Files API."""
    from app.services.gemini import upload_file_to_gemini

    mock_client = MagicMock()
    mock_file = MagicMock()
    mock_file.name = "files/dev-upload"
    mock_file.uri = "https://example.com/files/dev-upload"
    mock_client.files.upload.return_value = mock_file

    with (
        patch.dict("os.environ", {}, clear=True),
        patch("app.services.gemini.client", mock_client),
    ):
        result = upload_file_to_gemini(b"test data", "test.txt", "text/plain")

    assert result.name == "files/dev-upload"
    assert result.uri == "https://example.com/files/dev-upload"
    mock_client.files.upload.assert_called_once()


def test_upload_file_to_gemini_prod_path():
    """In production, upload_file_to_gemini uses GCS."""
    from app.services.gemini import upload_file_to_gemini

    with (
        patch.dict("os.environ", {"ENV": "production", "GCS_BUCKET": "prod-bucket"}),
        patch("app.services.gemini.upload_to_gcs") as mock_gcs,
    ):
        result = upload_file_to_gemini(b"test data", "report.pdf", "application/pdf")

    assert result.uri.startswith("gs://prod-bucket/uploads/")
    assert result.name.startswith("uploads/")
    mock_gcs.assert_called_once()


def test_delete_file_from_gemini_gcs_path():
    """delete_file_from_gemini routes to GCS when URI starts with gs://."""
    from app.services.gemini import delete_file_from_gemini

    with (
        patch("app.services.gemini.delete_from_gcs") as mock_gcs_del,
        patch("app.services.gemini.client.files.delete") as mock_gemini_del,
    ):
        delete_file_from_gemini(
            gemini_file_name="uploads/report_abc123.pdf",
            gemini_file_uri="gs://prod-bucket/uploads/report_abc123.pdf",
        )

    mock_gcs_del.assert_called_once_with("uploads/report_abc123.pdf")
    mock_gemini_del.assert_not_called()


def test_delete_file_from_gemini_files_api_path():
    """delete_file_from_gemini routes to Gemini Files API for files/ paths."""
    from app.services.gemini import delete_file_from_gemini

    with (
        patch("app.services.gemini.delete_from_gcs") as mock_gcs_del,
        patch("app.services.gemini.client.files.delete") as mock_gemini_del,
    ):
        delete_file_from_gemini(
            gemini_file_name="files/abc-123",
            gemini_file_uri="https://generativelanguage.googleapis.com/v1beta/files/abc-123",
        )

    mock_gemini_del.assert_called_once_with(name="files/abc-123")
    mock_gcs_del.assert_not_called()
