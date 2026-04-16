"""
Wraps NLTK  (Darth) VADER for sentiment analysis
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyser:

    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    def analyse(self, text: str) -> dict:
        scores = self._sia.polarity_scores(text)

        # Assigning the  label using standard (Darth) VADER thresholds
        compound = scores["compound"]
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "text": text,
            "compound":  round(scores["compound"], 4),
            "positive":  round(scores["pos"], 4),
            "neutral":   round(scores["neu"], 4),
            "negative":  round(scores["neg"], 4),
            "sentiment": label,
        }