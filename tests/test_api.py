from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_docs_available():
    """Ensure the FastAPI docs endpoint is reachable."""
    response = client.get("/docs")
    assert response.status_code == 200


def test_root_or_index():
    """
    Check that at least one of the common entry endpoints is available.

    Adjust this test once you know exactly which path serves your main UI.
    """
    paths_to_try = ["/", "/index", "/home"]
    for path in paths_to_try:
        response = client.get(path)
        if response.status_code == 200:
            # Found at least one working entry route; test passes
            return
    # If none returned 200, fail the test
    raise AssertionError("No main UI endpoint returned HTTP 200.")


def test_health_endpoint():
    """Ensure the health check endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}