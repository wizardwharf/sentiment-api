"""
analysis.py – Sentiment Analysis Module
========================================
Encapsulates the sentiment-analysis logic so that the Flask routes remain
thin and the model can be swapped out independently (e.g. replaced with a
transformer-based classifier) without touching the API layer.

The current implementation uses NLTK's **VADER** (Valence Aware Dictionary
and sEntiment Reasoner), a rule-based model specifically tuned for social-
media and short-form text.  VADER returns four scores:

* **compound** – normalised aggregate score in [-1, 1].
* **pos / neu / neg** – proportions of text that fall into each category
  (they sum to 1.0).

The compound score is used to assign an overall sentiment label using
standard thresholds recommended by VADER's authors:

    compound >=  0.05  →  positive
    compound <= -0.05  →  negative
    otherwise          →  neutral

Reference
---------
Hutto, C.J. & Gilbert, E.E. (2014). 'VADER: A Parsimonious Rule-based
Model for Sentiment Analysis of Social Media Text'. *Proceedings of the
Eighth International AAAI Conference on Weblogs and Social Media*.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download the VADER lexicon if it hasn't been downloaded yet.
# The `quiet=True` flag suppresses output after the first download.
nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyser:
    """
    Thin wrapper around NLTK's VADER sentiment analyser.

    Attributes
    ----------
    _sia : SentimentIntensityAnalyzer
        The underlying NLTK analyser instance.
    """

    def __init__(self):
        self._sia = SentimentIntensityAnalyzer()

    def analyse(self, text: str) -> dict:
        """
        Analyse the sentiment of *text* and return a result dictionary.

        Parameters
        ----------
        text : str
            The input text to analyse.

        Returns
        -------
        dict
            Keys: text, compound, positive, neutral, negative, sentiment.
        """
        scores = self._sia.polarity_scores(text)

        # Determine the overall label from the compound score.
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
