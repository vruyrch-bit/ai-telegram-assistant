import os


# --------------------------------------------------
# TEST ENVIRONMENT VARIABLES
# --------------------------------------------------

# These are fake values used only during tests.
# No real API keys or database passwords are needed.

os.environ["DATABASE_URL"] = (
    "postgresql://test:test@localhost:5432/test"
)

os.environ["ADMIN_API_KEY"] = "test-admin-key"


# --------------------------------------------------
# IMPORT APPLICATION
# --------------------------------------------------

from fastapi.testclient import TestClient

from api import app


client = TestClient(app)


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "AI Telegram Bot API"
    assert data["status"] == "running"


# --------------------------------------------------
# STATS MUST REQUIRE AUTHENTICATION
# --------------------------------------------------

def test_stats_requires_api_key():

    response = client.get("/stats")

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid or missing API key"
    }


# --------------------------------------------------
# USERS MUST REQUIRE AUTHENTICATION
# --------------------------------------------------

def test_users_requires_api_key():

    response = client.get("/users")

    assert response.status_code == 401

    assert response.json() == {
        "detail": "Invalid or missing API key"
    }


# --------------------------------------------------
# WRONG API KEY MUST FAIL
# --------------------------------------------------

def test_wrong_api_key():

    response = client.get(
        "/stats",
        headers={
            "X-API-Key": "wrong-key"
        }
    )

    assert response.status_code == 401


def test_activity_requires_api_key():
    assert client.get('/activity').status_code == 401
