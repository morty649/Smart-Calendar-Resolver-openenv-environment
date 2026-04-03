from fastapi.testclient import TestClient

from calender_en.server.app import app


def test_health_endpoint_returns_healthy() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}
