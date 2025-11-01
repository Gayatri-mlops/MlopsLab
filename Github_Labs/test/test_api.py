from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert "schemes" in r.json()

def test_convert_endpoint():
    r = client.post("/convert", json={"gpa": 3.6, "scheme":"us_4_linear"})
    assert r.status_code == 200
    assert abs(r.json()["percentage"] - 90.0) < 1e-6

def test_batch_endpoint():
    r = client.post("/batch", json={"gpas":[4.0, 3.0], "scheme":"us_4_linear"})
    assert r.status_code == 200
    assert r.json()["percentages"] == [100.0, 75.0]

def test_transcript_endpoint():
    payload = {
        "courses":[{"grade":"A", "credits":3}, {"grade":"B", "credits":3}],
        "grade_scale_max": 4.0,
        "scheme": "us_4_linear"
    }
    r = client.post("/transcript", json=payload)
    assert r.status_code == 200
    js = r.json()
    assert 0 <= js["weighted_gpa"] <= 4.0
    assert 0 <= js["percentage"] <= 100.0

def test_bad_input_error():
    r = client.post("/convert", json={"gpa": 4.5, "scheme":"us_4_linear"})
    assert r.status_code == 400