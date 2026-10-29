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
   Hand-labeled test set (~300 posts, slang flagged)
        │
   ┌────────┬─────────┬────────────┬──────────────┬────────────┐
 VADER   FinBERT  CryptoBERT  LLM zero/few-shot  LLM + LoRA
   └────────┴─────────┴────────────┴──────────────┴────────────┘
        │
   One evaluation script ──► metrics, slang vs plain, error analysis
        │
   Fine-tuned model hosted locally ──► latency
```

---

## Data

- **Training data:** [`ElKulako/stocktwits-crypto`](https://huggingface.co/datasets/ElKulako/stocktwits-crypto), StockTwits crypto posts (Nov 2021 – Jun 2022), MIT license.
- **Test set:** ~300 posts labeled by hand, each flagged as slang-heavy or plain.
- **Known caveat:** CryptoBERT was fine-tuned on posts from the same StockTwits dataset, so its scores may be optimistic. This is discussed in the results.

---

## Evaluation

- Macro-F1 (primary) and per-class precision / recall / F1
- Confusion matrices
- Accuracy on slang-heavy vs plain posts
- Agreement between authors' own tags and hand labels
- Invalid-output rate for the LLM (replies that aren't a valid label)
- Inference latency with the fine-tuned model running locally

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

Setup instructions will be added as the project is built.

---

## Status

🚧 In progress. See the stage plan in [`CLAUDE.md`](CLAUDE.md).

---

## Acknowledgments

- Prof. Joe Dumoulin, DATA 575, Gonzaga University
- Dataset and CryptoBERT by [ElKulako](https://huggingface.co/ElKulako)
- VADER (Hutto & Gilbert, 2014) and FinBERT (Araci, 2019)
- Built with help from Claude (Anthropic) as a coding assistant
