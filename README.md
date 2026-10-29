# SlangSense

**Fine-tuning small open-weight LLMs to read crypto sentiment and slang.**

Can a small, fine-tuned LLM read crypto slang better than standard sentiment tools?

Crypto social media has its own language: *HODL*, *FUD*, *rekt*, *WAGMI*, *rug pull*, *diamond hands*. General-purpose sentiment tools often misread it. This project fine-tunes a small open-weight language model (1–3B parameters) with LoRA to classify crypto posts as **bullish**, **bearish**, or **neutral**, and measures whether it beats general, finance, and crypto-specific baselines, especially on slang-heavy posts.

> DATA 575 (AI as Data Science: NLP to LLM), Gonzaga University · Final project · Work in progress

---

## Research question

Does LoRA fine-tuning a small open-weight LLM on crypto posts improve sentiment classification over:

1. a general rule-based tool (**VADER**),
2. a finance-domain model (**FinBERT**),
3. a crypto-domain model (**CryptoBERT**), and
4. the same LLM with **zero-shot** and **few-shot** prompting?

And does the gap change between **slang-heavy** and **plain** posts?

---

## Approach

| Model | Type | Role |
|---|---|---|
| VADER | Rule-based lexicon | General baseline |
| FinBERT | BERT fine-tuned on financial text | Finance baseline |
| CryptoBERT | BERTweet fine-tuned on crypto posts | Crypto baseline |
| LLM, zero-shot | Small instruct LLM, prompt only | Prompting baseline |
| LLM, few-shot | Same LLM + labeled examples in the prompt | Prompting baseline |
| **LLM + LoRA** | Same LLM, fine-tuned with (Q)LoRA | **Main model** |

Every model writes its predictions in the same format, and a single evaluation script scores all of them on the same hand-labeled test set.

```
StockTwits crypto posts
        │
   Data prep ──► train / val / held-out pool
        │
   Hand-labeled test set (150 slang-heavy + 150 plain posts)
        │
   ┌────────┬─────────┬────────────┬──────────────┬────────────┐
 VADER   FinBERT  CryptoBERT  LLM zero/few-shot  LLM + LoRA
   └────────┴─────────┴────────────┴──────────────┴────────────┘
        │
   One evaluation script ──► metrics, slang vs plain, error analysis
        │
   Fine-tuned model hosted locally ──► latency (optional)
```

---

## Data

- **Training data:** [`ElKulako/stocktwits-crypto`](https://huggingface.co/datasets/ElKulako/stocktwits-crypto), 1.33M StockTwits crypto posts (Nov 2021 – Jun 2022), MIT license. The labels are the authors' own tags: **bullish** or **bearish**, plus **neutral** for posts the author left untagged ([source](https://huggingface.co/datasets/ElKulako/stocktwits-crypto/discussions/3)). After removing duplicates, balanced samples are drawn for training (30k posts) and validation (3k).
- **Test set:** 300 posts from a separate held-out pool, labeled by hand: 150 slang-heavy and 150 plain. Slang is oversampled on purpose, so results are reported for each group separately. The author's tag is kept for comparison.
- **Known caveats:**
  - Author tags are noisy, and "neutral" really means "untagged", so models trained on these tags learn that noise. The hand-labeled test set measures real sentiment instead.
  - CryptoBERT was fine-tuned on posts from the same StockTwits dataset, so its scores may be optimistic.

---

## Evaluation

- Macro-F1 (primary) and per-class precision / recall / F1
- Confusion matrices
- Every metric reported separately for slang-heavy and plain posts
- How common slang is across the full dataset
- Agreement between authors' own tags and hand labels
- Invalid-output rate for the LLM (replies that aren't a valid label)
- Inference latency with the fine-tuned model running locally (optional)

---

## Project structure

```
SlangSense/
├── data/          # processed splits, hand-labeled test set, slang list (raw data not committed)
├── notebooks/     # one notebook per stage
├── src/           # shared code: label mapping, output parsing, evaluation
├── results/       # predictions, metrics, figures
└── README.md
```

---

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11 (uv can install it for you).

```bash
git clone <this repo's URL>
cd SlangSense

# Optional: keeps the environment out of iCloud sync if the folder is on a synced Mac Desktop.
# Skip this line to use uv's default .venv folder.
export UV_PROJECT_ENVIRONMENT=venv.nosync

uv sync
```

Open the notebooks in VS Code, select the environment's Python (`venv.nosync` or `.venv`) as the kernel, and run them in order:

| Notebook | What it does | Runs on |
|---|---|---|
| `01_data_prep.ipynb` | Downloads and cleans the dataset; writes the train / val / test-pool splits | Mac, about 1 minute |

---

## Status

🚧 In progress. See the stage plan in [`CLAUDE.md`](CLAUDE.md).

---

## Acknowledgments

- Prof. Joe Dumoulin, DATA 575, Gonzaga University
- Dataset and CryptoBERT by [ElKulako](https://huggingface.co/ElKulako)
- VADER (Hutto & Gilbert, 2014) and FinBERT (Araci, 2019)
- Built with help from Claude (Anthropic) as a coding assistant
