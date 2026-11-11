"""Label names and mappings shared by every notebook."""

# The three labels every model must predict.
LABELS = ["bullish", "bearish", "neutral"]

# What a model's prediction becomes when its output can't be read as a label.
# The evaluation counts it as wrong.
INVALID = "INVALID"

# ElKulako/stocktwits-crypto stores labels as integers, with the same coding
# as CryptoBERT's config. Authors tag posts Bullish or Bearish on StockTwits;
# posts with no tag were stored as neutral (confirmed by the dataset author).
DATASET_LABELS = {0: "bearish", 1: "neutral", 2: "bullish"}

# FinBERT (ProsusAI/finbert) predicts financial sentiment.
FINBERT_LABELS = {"positive": "bullish", "negative": "bearish", "neutral": "neutral"}

# CryptoBERT (ElKulako/cryptobert) already uses our three classes, capitalized.
CRYPTOBERT_LABELS = {"Bullish": "bullish", "Bearish": "bearish", "Neutral": "neutral"}


def vader_label(compound):
    """Turn a VADER compound score (-1 to 1) into a label, using VADER's standard cutoffs."""
    if compound >= 0.05:
        return "bullish"
    if compound <= -0.05:
        return "bearish"
    return "neutral"
