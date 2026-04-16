"""
Sentiment Analysis REST API
============================
A Flask-based REST API that uses NLTK's VADER sentiment analyser to classify
the emotional tone of input text.  Designed for organisations to understand
feedback from service users.

Endpoints
---------
POST /analyse   – Submit text and receive a sentiment classification.
GET  /results   – Retrieve the average sentiment scores across all requests.
DELETE /results  – Clear all stored results (useful for testing / resets).
GET  /health    – Simple health-check endpoint.

Storage: all request data is held **in-memory only** and is lost when the
server process stops (as required by the specification).
"""

from flask import Flask, request, jsonify
from analysis import SentimentAnalyser

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app():
    """
    Application factory pattern – makes the app easier to test and
    configure independently of module-level state.
    """
    app = Flask(__name__)

    # Shared, in-memory store for sentiment results.
    # Each item is a dict produced by SentimentAnalyser.analyse().
    results_store: list[dict] = []

    # Instantiate the analyser once at startup so the VADER lexicon is
    # loaded only once rather than on every request.
    analyser = SentimentAnalyser()

    # -------------------------------------------------------------------
    # Routes
    # -------------------------------------------------------------------

    @app.route("/analyse", methods=["POST"])
    def analyse():
        """
        Analyse the sentiment of submitted text.

        Expects JSON: {"text": "<string to analyse>"}
        Returns JSON with compound, positive, neutral, negative scores
        and an overall label (positive / negative / neutral).
        """
        # --- Input validation ------------------------------------------------
        if not request.is_json:
            return jsonify({
                "error": "Request must include Content-Type: application/json"
            }), 415  # Unsupported Media Type

        data = request.get_json(silent=True)
        if data is None or "text" not in data:
            return jsonify({
                "error": "Request body must contain a 'text' field"
            }), 400  # Bad Request

        text = data["text"]

        if not isinstance(text, str) or text.strip() == "":
            return jsonify({
                "error": "'text' must be a non-empty string"
            }), 400

        # --- Analysis --------------------------------------------------------
        result = analyser.analyse(text.strip())
        results_store.append(result)

        return jsonify(result), 200

    @app.route("/results", methods=["GET"])
    def results():
        """
        Return the average sentiment scores for every request made so far.

        If no requests have been recorded yet, returns a message indicating
        that no data is available.
        """
        if len(results_store) == 0:
            return jsonify({
                "message": "No analysis requests have been made yet.",
                "count": 0
            }), 200

        count = len(results_store)
        avg_compound = sum(r["compound"] for r in results_store) / count
        avg_positive = sum(r["positive"] for r in results_store) / count
        avg_neutral  = sum(r["neutral"]  for r in results_store) / count
        avg_negative = sum(r["negative"] for r in results_store) / count

        # Derive an overall label from the average compound score using
        # the same thresholds as individual analyses.
        if avg_compound >= 0.05:
            overall_label = "positive"
        elif avg_compound <= -0.05:
            overall_label = "negative"
        else:
            overall_label = "neutral"

        return jsonify({
            "count": count,
            "average_scores": {
                "compound": round(avg_compound, 4),
                "positive": round(avg_positive, 4),
                "neutral":  round(avg_neutral, 4),
                "negative": round(avg_negative, 4)
            },
            "overall_sentiment": overall_label
        }), 200

    @app.route("/results", methods=["DELETE"])
    def clear_results():
        """Clear all stored results.  Useful for testing and resets."""
        results_store.clear()
        return jsonify({"message": "All results have been cleared."}), 200

    @app.route("/health", methods=["GET"])
    def health():
        """Lightweight health-check endpoint."""
        return jsonify({"status": "healthy"}), 200

    return app


# ---------------------------------------------------------------------------
# Development server entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)
