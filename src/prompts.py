"""The LLM prompt and how replies are generated.

Zero-shot, few-shot, the fine-tune (Stage 5) and local inference (Stage 7) all use this file,
so every LLM result comes from the same prompt and the same decoding.
"""

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


def use_greedy_decoding(model):
    """Make the model always pick the most likely next token, so replies repeat exactly across runs."""
    settings = model.generation_config
    settings.do_sample = False  # Qwen samples randomly by default
    settings.temperature = None  # temperature, top_p and top_k only matter when sampling
    settings.top_p = None
    settings.top_k = None
    settings.repetition_penalty = 1.0  # off: it would push the model away from label words already in the prompt
    settings.max_new_tokens = 10  # a label needs about 3; the rest shows what an off-format reply says


def ask_llm(model, tokenizer, text, few_shot=False):
    """Ask the LLM to label one post. Returns its raw reply."""
    messages = build_messages(text, few_shot)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]  # generate returns the prompt + the reply
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
