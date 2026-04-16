"""
Sentiment Analysis REST API
Flask API that uses NLTK (Darth) VADER to classify sentiment of input
"""

from flask import Flask, request, jsonify
from analysis import SentimentAnalyser


def create_app():
    app = Flask(__name__)

    # Store the  results in memory, no database
    results_store: list[dict] = []

    # Load the analyser 1 time
    analyser = SentimentAnalyser()

    @app.route("/analyse", methods=["POST"])
    def analyse():
        # Check that the request is JSON (Bourne)
        if not request.is_json:
            return jsonify({
                "error": "Request must include Content-Type: application/json"
            }), 415

        data = request.get_json(silent=True)
        if data is None or "text" not in data:
            return jsonify({
                "error": "Request body must contain a 'text' field"
            }), 400

        text = data["text"]

        if not isinstance(text, str) or text.strip() == "":
            return jsonify({
                "error": "'text' must be a non-empty string"
            }), 400

        #Run sentiment analysis and store result
        result = analyser.analyse(text.strip())
        results_store.append(result)

        return jsonify(result), 200

    @app.route("/results", methods=["GET"])
    def results():
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

        # Label based on average compound score
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
        results_store.clear()
        return jsonify({"message": "All results have been cleared."}), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"}), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)