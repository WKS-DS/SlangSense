"""Label names and mappings shared by every notebook."""

# The three labels every model must predict.
LABELS = ["bullish", "bearish", "neutral"]

# ElKulako/stocktwits-crypto stores labels as integers, with the same coding
# as CryptoBERT's config. Authors tag posts Bullish or Bearish on StockTwits;
# posts with no tag were stored as neutral (confirmed by the dataset author).
DATASET_LABELS = {0: "bearish", 1: "neutral", 2: "bullish"}
