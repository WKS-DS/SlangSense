"""The one scoring path: every model's predictions are scored here, on the same test set.

Each model writes results/preds_<model>.csv with two columns, post_id and pred_label.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.labels import INVALID, LABELS, parse_reply

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "test_hand_labeled.csv"
RESULTS_DIR = ROOT / "results"

# Results are reported for all posts, and separately by the hand slang flag.
GROUPS = {"all": None, "slang-heavy": 1, "plain": 0}


def load_test():
    """The 300 hand-labeled test posts."""
    return pd.read_csv(TEST_PATH)


def save_preds(post_ids, pred_labels, model):
    """Write results/preds_<model>.csv, the format every model uses."""
    preds = pd.DataFrame({"post_id": post_ids, "pred_label": pred_labels})
    RESULTS_DIR.mkdir(exist_ok=True)
    preds.to_csv(RESULTS_DIR / f"preds_{model}.csv", index=False)


def save_replies(post_ids, replies, model):
    """For an LLM: save its raw replies (for error analysis) and the labels parsed from them (for scoring)."""
    pred_labels = [parse_reply(reply) for reply in replies]
    pd.DataFrame({"post_id": post_ids, "reply": replies, "pred_label": pred_labels}).to_csv(
        RESULTS_DIR / f"replies_{model}.csv", index=False
    )
    save_preds(post_ids, pred_labels, model)


def load_preds(model, test):
    """Read a model's predictions and check there is exactly one valid prediction per test post."""
    preds = pd.read_csv(RESULTS_DIR / f"preds_{model}.csv")

    assert preds["post_id"].is_unique, f"{model}: some posts have more than one prediction"
    missing = set(test["post_id"]) - set(preds["post_id"])
    extra = set(preds["post_id"]) - set(test["post_id"])
    assert not missing and not extra, f"{model}: {len(missing)} test posts missing, {len(extra)} unknown posts"

    bad = set(preds["pred_label"]) - set(LABELS) - {INVALID}
    assert not bad, f"{model}: unknown labels {bad}"
    return preds


def confusion_matrix(true, pred):
    """Count posts by true label (rows) and predicted label (columns). INVALID gets its own column."""
    return pd.crosstab(
        pd.Categorical(true, categories=LABELS),
        pd.Categorical(pred, categories=LABELS + [INVALID]),
        rownames=["true"],
        colnames=["predicted"],
        dropna=False,  # keep every label, even one that was never predicted
    )


def per_class_scores(matrix):
    """Precision, recall and F1 for each label, read off the confusion matrix."""
    rows = []
    for label in LABELS:
        correct = matrix.loc[label, label]
        predicted = matrix[label].sum()  # posts the model called this label
        actual = matrix.loc[label].sum()  # posts that truly have this label

        precision = correct / predicted if predicted else 0.0
        recall = correct / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"label": label, "precision": precision, "recall": recall, "f1": f1, "posts": actual})
    return pd.DataFrame(rows).set_index("label")


def score_posts(true, pred):
    """All the scores for one set of posts."""
    true, pred = list(true), list(pred)
    matrix = confusion_matrix(true, pred)
    per_class = per_class_scores(matrix)
    return {
        "posts": len(true),
        "macro_f1": per_class["f1"].mean(),  # every label counts equally, however rare
        "accuracy": sum(t == p for t, p in zip(true, pred)) / len(true),
        "invalid_rate": pred.count(INVALID) / len(pred),
        "per_class": per_class,
        "confusion": matrix,
    }


def test_with_preds(model):
    """The test posts with the model's prediction added as a pred_label column."""
    test = load_test()
    return test.merge(load_preds(model, test), on="post_id")


def select_group(posts, group):
    """All posts, or only the slang-heavy or plain ones (by the hand slang flag)."""
    flag = GROUPS[group]
    return posts if flag is None else posts[posts["slang"] == flag]


def evaluate(model, truth="label"):
    """Score results/preds_<model>.csv on all posts, slang-heavy posts and plain posts.

    truth is the column the predictions are compared with: "label" (the hand labels, the
    default) or "author_label" (the authors' own tags, to see which models learned them).
    """
    merged = test_with_preds(model)
    results = {}
    for group in GROUPS:
        posts = select_group(merged, group)
        results[group] = score_posts(posts[truth], posts["pred_label"])
    return results


def bootstrap_macro_f1(model, group="all", resamples=1000, seed=42):
    """Macro-F1 on 1,000 resamples of the test posts, each drawn at random with replacement.

    The spread of these scores shows how much a score could move with a different sample
    of 300 posts. The same seed draws the same resamples for every model, so two models
    can be compared resample by resample.
    """
    posts = select_group(test_with_preds(model), group)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(resamples):
        sample = posts.sample(len(posts), replace=True, random_state=rng)
        scores.append(score_posts(sample["label"], sample["pred_label"])["macro_f1"])
    return np.array(scores)


def score_table(models, truth="label"):
    """One row per model and group, with the headline numbers."""
    rows = []
    for model in models:
        for group, scores in evaluate(model, truth).items():
            rows.append({
                "model": model,
                "group": group,
                "posts": scores["posts"],
                "macro_f1": scores["macro_f1"],
                "accuracy": scores["accuracy"],
                "invalid_rate": scores["invalid_rate"],
            })
    return pd.DataFrame(rows)
