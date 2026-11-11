"""The one scoring path: every model's predictions are scored here, on the same test set.

Each model writes results/preds_<model>.csv with two columns, post_id and pred_label.
"""

from pathlib import Path

import pandas as pd

from src.labels import INVALID, LABELS

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


def evaluate(model):
    """Score results/preds_<model>.csv on all posts, slang-heavy posts and plain posts."""
    test = load_test()
    merged = test.merge(load_preds(model, test), on="post_id")

    results = {}
    for group, flag in GROUPS.items():
        posts = merged if flag is None else merged[merged["slang"] == flag]
        results[group] = score_posts(posts["label"], posts["pred_label"])
    return results


def score_table(models):
    """One row per model and group, with the headline numbers."""
    rows = []
    for model in models:
        for group, scores in evaluate(model).items():
            rows.append({
                "model": model,
                "group": group,
                "posts": scores["posts"],
                "macro_f1": scores["macro_f1"],
                "accuracy": scores["accuracy"],
                "invalid_rate": scores["invalid_rate"],
            })
    return pd.DataFrame(rows)
