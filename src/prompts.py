"""The LLM prompt. Zero-shot, few-shot, the fine-tune (Stage 5) and local inference (Stage 7) all use it."""

SYSTEM_PROMPT = "You classify the sentiment of crypto social media posts."

# The label definitions follow the hand-labeling guide in notebook 02,
# so the model is asked the same question the test set answers.
INSTRUCTIONS = """Classify this crypto post as bullish, bearish or neutral.

- bullish: the author expects or wants the price to rise, is buying or holding, or celebrates gains.
- bearish: the author expects the price to fall, is selling or shorting, warns others, or calls a coin a scam.
- neutral: no clear lean, such as news, facts, plain questions, greetings, ads or spam.

Post: {text}

Answer with one word: bullish, bearish or neutral."""

# Few-shot examples: 2 per class from train.csv, one with slang and one without.
# They are mixed so the order gives no hint, and the same 6 are used for every post.
FEW_SHOT_EXAMPLES = [
    {"post_id": 1079812, "label": "bullish", "text": "fud will come and go, doge is here to stay."},
    {"post_id": 1117822, "label": "bearish", "text": "all faith is gone. crypto to be worthless."},
    {"post_id": 1217163, "label": "neutral", "text": "new article : ethereum classic – this may be the answer to the question ‘to hodl or sell'"},
    {"post_id": 1310880, "label": "bearish", "text": "biggest rug pull in history of financial markets incoming 🤘🏼"},
    {"post_id": 396344, "label": "neutral", "text": "has anyone successfully sent cro from coin base to the cro app?"},
    {"post_id": 157872, "label": "bullish", "text": "first green candle all day…keep them coming"},
]


def build_messages(text, few_shot=False):
    """The chat messages for one post. Few-shot adds the examples as earlier questions with correct answers."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if few_shot:
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": INSTRUCTIONS.format(text=example["text"])})
            messages.append({"role": "assistant", "content": example["label"]})
    messages.append({"role": "user", "content": INSTRUCTIONS.format(text=text)})
    return messages
