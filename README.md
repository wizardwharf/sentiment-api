# Sentiment Analysis REST API

A Flask-based REST API that uses NLTK's VADER sentiment analyser to classify the emotional tone of input text. Designed for organisations to process and understand feedback left by service users.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Setup Instructions](#setup-instructions)
- [Running the Server](#running-the-server)
- [API Documentation](#api-documentation)
- [Running Tests](#running-tests)
- [Design Decisions](#design-decisions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   Client                        │
│          (curl / Postman / Frontend)            │
└──────────────────┬──────────────────────────────┘
                   │  HTTP (JSON)
                   ▼
┌─────────────────────────────────────────────────┐
│              Flask Application                  │
│                  (app.py)                       │
│                                                 │
│  POST /analyse ──► SentimentAnalyser.analyse()  │
│  GET  /results ──► In-memory results store      │
│  DELETE /results ─► Clear results store          │
│  GET  /health  ──► Status check                 │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│           analysis.py                           │
│   SentimentAnalyser (wraps NLTK VADER)          │
│                                                 │
│   • Loads VADER lexicon once at startup         │
│   • Returns compound, pos, neu, neg scores      │
│   • Assigns overall label via compound threshold│
└─────────────────────────────────────────────────┘
```

The application follows a **layered architecture** with clear separation of concerns:

| Layer | File | Responsibility |
|-------|------|----------------|
| **API / Routing** | `app.py` | HTTP request handling, input validation, JSON responses |
| **Analysis** | `analysis.py` | Sentiment model logic (VADER), score computation |
| **Storage** | In-memory `list` in `app.py` | Holds results for the lifetime of the server process |

---

## Setup Instructions

### Prerequisites

- Python 3.10 or later
- pip (Python package manager)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/sentiment-api.git
cd sentiment-api

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

The VADER lexicon will be downloaded automatically on first run.

---

## Running the Server

```bash
python app.py
```

The server starts on `http://127.0.0.1:5000` by default (debug mode enabled for development).

---

## API Documentation

### Base URL

```
http://127.0.0.1:5000
```

All request and response bodies use **JSON** (`Content-Type: application/json`).

---

### `POST /analyse`

Submit text for sentiment analysis.

#### Request

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | The text to analyse. Must be non-empty. |

#### Example Request

```bash
curl -X POST http://127.0.0.1:5000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "The staff were incredibly helpful and friendly!"}'
```

#### Example Response (200 OK)

```json
{
  "text": "The staff were incredibly helpful and friendly!",
  "compound": 0.7351,
  "positive": 0.525,
  "neutral": 0.475,
  "negative": 0.0,
  "sentiment": "positive"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | The original input text |
| `compound` | float | Normalised aggregate score (−1.0 to 1.0) |
| `positive` | float | Proportion of text classified as positive (0–1) |
| `neutral` | float | Proportion of text classified as neutral (0–1) |
| `negative` | float | Proportion of text classified as negative (0–1) |
| `sentiment` | string | Overall label: `"positive"`, `"negative"`, or `"neutral"` |

#### Error Responses

| Status | Condition | Example Body |
|--------|-----------|--------------|
| 400 | Missing or empty `text` field | `{"error": "Request body must contain a 'text' field"}` |
| 415 | Non-JSON content type | `{"error": "Request must include Content-Type: application/json"}` |

---

### `GET /results`

Retrieve the average sentiment scores across all `POST /analyse` requests made since the server started (or since the last `DELETE /results`).

#### Example Request

```bash
curl http://127.0.0.1:5000/results
```

#### Example Response (200 OK – with data)

```json
{
  "count": 42,
  "average_scores": {
    "compound": 0.3214,
    "positive": 0.2871,
    "neutral": 0.5932,
    "negative": 0.1197
  },
  "overall_sentiment": "positive"
}
```

#### Example Response (200 OK – no data yet)

```json
{
  "message": "No analysis requests have been made yet.",
  "count": 0
}
```

---

### `DELETE /results`

Clear all stored results. Useful for resetting between test runs.

#### Example Request

```bash
curl -X DELETE http://127.0.0.1:5000/results
```

#### Example Response (200 OK)

```json
{
  "message": "All results have been cleared."
}
```

---

### `GET /health`

Simple health-check endpoint.

#### Example Request

```bash
curl http://127.0.0.1:5000/health
```

#### Example Response (200 OK)

```json
{
  "status": "healthy"
}
```

---

## Python Client Example

```python
import requests

BASE = "http://127.0.0.1:5000"

# Analyse some feedback
feedbacks = [
    "Excellent service, very impressed!",
    "The waiting time was far too long.",
    "It was okay, nothing special.",
]

for fb in feedbacks:
    resp = requests.post(f"{BASE}/analyse", json={"text": fb})
    result = resp.json()
    print(f"{result['sentiment']:>8}  ({result['compound']:+.4f})  {fb}")

# Fetch averages
averages = requests.get(f"{BASE}/results").json()
print(f"\nAverage compound score over {averages['count']} requests: "
      f"{averages['average_scores']['compound']:+.4f} "
      f"({averages['overall_sentiment']})")
```

---

## Running Tests

```bash
pytest tests/ -v
```

All tests use Flask's built-in test client and require no running server.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **NLTK VADER** over transformer models | Lightweight, no GPU required, well-suited to short feedback text. Can be swapped for a Hugging Face transformer by changing only `analysis.py`. |
| **Application factory pattern** (`create_app()`) | Industry best practice for Flask; enables independent test instances and avoids shared global state between tests. |
| **In-memory list** for storage | Meets the specification requirement of non-persistent storage. Simple and thread-safe for development use. |
| **Separate `analysis.py` module** | Decouples ML logic from HTTP handling, following the Single Responsibility Principle. Makes it straightforward to unit-test the analyser or replace the model. |
| **Standard VADER thresholds** (±0.05) | Recommended by the model's original authors (Hutto & Gilbert, 2014). |

---

## References

Hutto, C.J. and Gilbert, E.E. (2014) 'VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text', *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media*. Ann Arbor, MI: AAAI Press.
