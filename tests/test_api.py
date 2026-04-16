"""Tests for the  API"""

import pytest
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "healthy"


class TestAnalyse:
    def test_positive_text(self, client):
        resp = client.post("/analyse", json={"text": "I love this service! It's absolutely fantastic."})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["sentiment"] == "positive"
        assert data["compound"] > 0

    def test_negative_text(self, client):
        resp = client.post("/analyse", json={"text": "This is terrible. I am very disappointed."})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["sentiment"] == "negative"
        assert data["compound"] < 0

    def test_neutral_text(self, client):
        resp = client.post("/analyse", json={"text": "The meeting is at 3pm today."})
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["sentiment"] == "neutral"

    def test_missing_text_field(self, client):
        resp = client.post("/analyse", json={"message": "hello"})
        assert resp.status_code == 400

    def test_empty_text(self, client):
        resp = client.post("/analyse", json={"text": ""})
        assert resp.status_code == 400

    def test_non_json_request(self, client):
        resp = client.post("/analyse", data="hello", content_type="text/plain")
        assert resp.status_code == 415

    def test_response_contains_all_keys(self, client):
        resp = client.post("/analyse", json={"text": "Good work."})
        data = resp.get_json()
        for key in ("text", "compound", "positive", "neutral", "negative", "sentiment"):
            assert key in data


class TestResults:
    def test_empty_results(self, client):
        resp = client.get("/results")
        data = resp.get_json()
        assert resp.status_code == 200
        assert data["count"] == 0

    def test_average_after_multiple_requests(self, client):
        texts = [
            "I love this!",
            "This is awful.",
            "The office opens at 9am.",
        ]
        for t in texts:
            client.post("/analyse", json={"text": t})

        resp = client.get("/results")
        data = resp.get_json()
        assert data["count"] == 3
        assert "average_scores" in data
        assert "overall_sentiment" in data
        assert isinstance(data["average_scores"]["compound"], float)


class TestClearResults:
    def test_clear_resets_count(self, client):
        client.post("/analyse", json={"text": "Great job!"})
        client.delete("/results")
        resp = client.get("/results")
        assert resp.get_json()["count"] == 0