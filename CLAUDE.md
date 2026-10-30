# CLAUDE.md — SlangSense

Instructions for Claude Code. Read this whole file before doing any work in this repo.

## Who you're working with

- **Will Scott**, Data Science MS student at Gonzaga. Codes in Python at a **beginner-to-intermediate** level.
- Works in **VS Code** on a Mac (Apple Silicon). Fine-tuning runs on **Google Colab (T4 GPU)**.
- This is Will's **individual final project for DATA 575** (AI as Data Science: NLP to LLM). The professor approved the scope below. **Do not expand it.**

---

## 1. Project architecture

**Goal:** LoRA fine-tune a small open-weight LLM (1–3B) to classify crypto social media posts as `bullish`, `bearish`, or `neutral`, and compare it against baselines, with a focus on crypto slang.

**Core rule:** every model writes predictions to `results/preds_<model>.csv` with columns `post_id, pred_label`. **One** evaluation module scores every prediction file on the **same** hand-labeled test set. Never write a separate scoring path for one model.

### Pipeline

1. **Data prep.** Load `ElKulako/stocktwits-crypto` from Hugging Face. Map labels to `bullish / bearish / neutral`. Drop duplicates and very short posts. Build a balanced training subset (about 20–50k posts; the full dataset isn't needed). Make a validation split and a held-out pool for the test set. First, confirm how labels were made and what "neutral" means (the dataset card doesn't say).
2. **Test set.** 300 posts from the held-out pool: **150 slang-heavy + 150 plain**, each roughly balanced across bullish / bearish / neutral. Posts are pre-sorted with an automatic flag that matches terms in `data/slang.csv` (`term, meaning`; about 100 lowercase terms including variants; Claude drafts it and Will edits it). Will hand-labels both the **sentiment** and the **slang flag** for all 300 posts. The author's original tag is kept in a separate column for comparison. Also report the **natural slang rate** in the full dataset, since the test set oversamples slang.
3. **Baselines.**
   - VADER: compound score ≥ 0.05 → bullish, ≤ −0.05 → bearish, else neutral.
   - FinBERT (`ProsusAI/finbert`): positive / negative / neutral → bullish / bearish / neutral.
   - CryptoBERT (`ElKulako/cryptobert`): native Bearish / Neutral / Bullish.
4. **LLM prompting.** Small instruct model (e.g., Qwen2.5-1.5B/3B-Instruct or Llama-3.2-1B/3B-Instruct). The prompt asks for **one word**. Zero-shot, then few-shot with a **fixed** set of examples (2 per class) used for every post. A parser maps the reply to a label or `INVALID`.
5. **LoRA fine-tune.** On Colab T4 with QLoRA (4-bit) using Hugging Face `peft` + `trl`. Train prompt → label word. Save only the adapter. Predict on the test set.
6. **Evaluate.** Macro-F1 (primary), per-class P/R/F1, confusion matrix, author tag vs hand label agreement, invalid-output rate. Report results **separately for slang-heavy and plain posts**.
7. **Local hosting (optional).** Only if time allows after evaluation. Run the fine-tuned model on the Mac (Ollama / LM Studio via GGUF, MLX, or transformers on MPS; pick the simplest one that works). Measure latency per post.

### Known caveats (keep these visible in results)

- CryptoBERT was fine-tuned on 2M posts from the **same** StockTwits dataset, so its test score is likely optimistic.
- StockTwits labels are the authors' own tags, and "neutral" means the author left the post **untagged** (confirmed by the dataset author). So the training labels are noisy (sarcasm, hype, untagged posts that still carry an opinion).
- The dataset was lowercased, with cashtags, usernames, and URLs removed.
- The test set is 50% slang-heavy by design, far above the natural rate. That's why results are reported per group: a single combined score over-represents slang.

### Repo layout

```
SlangSense/
├── README.md                    # public-facing
├── CLAUDE.md                    # this file
├── pyproject.toml, uv.lock      # environment (uv)
├── .gitignore
├── data/
│   ├── processed/               # train.csv, val.csv, test_pool.csv
│   ├── interim.nosync/          # full cleaned dataset (not committed)
│   ├── slang.csv                # slang terms + meanings
│   ├── test_to_label.csv        # labeling sheet Will fills in (no author tag)
│   └── test_hand_labeled.csv    # Will's labels + author tag (the test set)
├── notebooks/                   # one notebook per stage, numbered (01_..., 02_...)
├── src/
│   ├── labels.py                # label mapping + LLM reply parser
│   ├── prompts.py               # shared LLM prompt + fixed few-shot examples
│   └── evaluate.py              # scores any preds CSV
├── models.nosync/               # LoRA adapter (not committed)
└── results/                     # preds_*.csv, metrics.csv, figures/
```

`src/prompts.py` was added because Stages 4, 5 and 7 must use the exact same prompt, and the Colab notebook imports it from the repo.

Adjust the layout only if there's a clear reason, and update this file when you do.

---

## 2. Keep it simple

- **Simple, readable code over clever code.** Explicit loops and clear variable names are fine. No unnecessary classes, abstractions, config systems, or frameworks.
- **Notebooks for steps, `src/` for anything reused** (label mapping, parsing, evaluation). Don't duplicate logic across notebooks.
- **Small, focused functions** with short docstrings. Comment the *why*, not the obvious *what*.
- **Fewest dependencies possible.** Add a package only when it's needed, and say why.
- **Don't build ahead.** Only build what the current stage needs.
- **Explain new concepts** (LoRA, quantization, macro-F1, etc.) in plain language the first time they appear.
- Use a fixed random seed (`42`) everywhere randomness matters, so results are reproducible.
- Keep notebook outputs small. No embedded charts with tens of thousands of points, and no printing whole datasets. Large outputs made an earlier assignment's notebook too big to upload.

### Environment notes

- Use **uv** (`uv sync`, `uv run`). Python 3.11.
- This folder is on the Desktop, which **syncs to iCloud**. iCloud hides files inside dot-folders and uploads everything. Name generated or heavy folders with a `.nosync` ending (e.g., the virtual environment `venv.nosync` via `UV_PROJECT_ENVIRONMENT=venv.nosync`, `data/raw.nosync/`, `models.nosync/`) and gitignore them. Will uses the same convention in his EmberEye project.
- `bitsandbytes` (4-bit) needs CUDA, so the QLoRA notebook runs on **Colab**, not the Mac. Everything else should run locally.
- Never commit API keys or tokens. Use environment variables if one is ever needed.

---

## 3. Work in stages

Before writing any code:

1. Review the architecture above and **decide how many stages the project needs**. Each stage should be small enough to finish in one or a few sessions and end with something that runs.
2. Write the plan into the **Stage plan** section below. Each stage gets a goal, its deliverables (files), and a "done when" check.
3. **Show Will the plan and wait for approval** before starting Stage 1.

While working:

- Work on **one stage at a time**. Don't start the next stage until the current one is done and Will has confirmed.
- When a stage is finished, update its checkbox and add a 2–4 sentence **plain-language summary** of what was built and why (this doubles as presentation notes).
- If something in the plan turns out wrong, say so, propose the change, and update this file.

### Stage plan

Seven stages: Stages 1–6 are required and Stage 7 is optional. Each stage gets its own branch (`stage-N-<name>`).

**Every stage ends the same way:** notebooks run top to bottom → README "Getting started" covers the new step → checkbox ticked and summary added below → Will confirms → merge into `main`, push, delete the branch.

- [x] Stage 1: Setup and data prep
- [ ] Stage 2: Slang list and hand-labeled test set
- [ ] Stage 3: Evaluation module and classic baselines
- [ ] Stage 4: LLM prompting (zero-shot and few-shot)
- [ ] Stage 5: QLoRA fine-tune (Colab)
- [ ] Stage 6: Final evaluation and write-up
- [ ] Stage 7 (optional): Local hosting and latency

#### Stage 1: Setup and data prep (`stage-1-data-prep`)

- **Goal:** A working repo and environment, plus clean train / val / test-pool splits, with the meaning of the labels confirmed.
- **Deliverables:**
  - Git repo, `.gitignore`, `pyproject.toml` + `uv.lock` (Python 3.11, environment in `venv.nosync`), folder skeleton.
  - `src/labels.py`: maps the dataset's labels to `bullish / bearish / neutral`.
  - `notebooks/01_data_prep.ipynb`:
    - Loads the dataset and works out how the labels were made and what "neutral" means, using the dataset card, the CryptoBERT card/paper and sample posts for each label.
    - Gives each post a stable `post_id`, then drops duplicates and very short posts.
    - Sets aside the held-out pool **first**, then draws a balanced training set (about 30k, 10k per class) and a validation set (about 3k).
  - `data/processed/train.csv`, `val.csv`, `test_pool.csv`. The pool holds about 15k posts, balanced by author tag; Stage 2 checks it has enough slang posts.
  - The full cleaned dataset saved to `data/interim.nosync/` (not committed), for the slang-rate count in Stage 2.
- **Packages:** `huggingface_hub` (download), `openpyxl` (the dataset is an Excel file), `pandas`, `ipykernel` (notebooks in VS Code). `datasets` and `scikit-learn` turned out not to be needed yet.
- **Done when:** the notebook runs top to bottom; split sizes and class counts match the targets; no post text appears in two splits; the notebook explains in plain words what "neutral" means. Check the GitHub account and **ask Will before the first push**.
- **Watch for:** if "neutral" turns out to mean "the author didn't tag it", the training neutral class is noisy. Flag it and decide with Will before moving on.
- **Summary:** Set up the repo, the uv environment and `.gitignore`. Notebook 01 downloads the 1.33M-post dataset (one Excel file with two sheets) and removes 151 empty rows, about 39k exact duplicates and 37k near duplicates (bot reposts that only change a price). It then draws balanced splits: 30k train, 3k val and a 15k test pool, with the pool set aside first so nothing leaks into the test set. The dataset author confirmed that labels are the authors' own tags and that "neutral" means untagged. That makes the training labels noisy, and it's the main reason the test set is labeled by hand.

#### Stage 2: Slang list and hand-labeled test set (`stage-2-test-set`)

- **Goal:** The 300-post hand-labeled test set (150 slang-heavy + 150 plain) and the natural slang rate.
- **Deliverables:**
  - `data/slang.csv` (`term, meaning`; about 100 lowercase terms including variants, e.g. `hodl`, `hodling`, `hodler`). Claude drafts it and Will reviews and edits it before any sampling.
  - `notebooks/02_test_set.ipynb`:
    - Automatic slang flag: whole-word and whole-phrase matching against `slang.csv`.
    - Natural slang rate: the share of posts in the full cleaned dataset with at least one slang term, overall and per class.
    - Sampling: 50 posts per (auto slang flag × author tag) group from `test_pool.csv`, which gives 300 posts, shuffled.
    - Labeling sheet `data/test_to_label.csv`: `post_id, text, matched_terms`, plus empty `slang` (1/0) and `label` columns. The author's tag is left out so Will labels blind.
    - A short labeling guide covering bullish / bearish / neutral, what counts as slang-heavy, and edge cases (sarcasm, questions, spam).
    - After Will labels: checks the sheet, joins the author's tag back by `post_id`, and writes `data/test_hand_labeled.csv`. It also reports the final group sizes, the mix of hand labels, and how often Will agreed with the automatic slang flag.
- **Done when:** `test_hand_labeled.csv` has 300 rows with valid labels and flags, and the natural slang rate and the auto-vs-hand flag agreement are reported.
- **Note:** posts are sampled by the automatic flag and author tag, but Will's slang flag and label are final. Group sizes and class balance may therefore move a little away from 150/150 and even classes. Report the actual counts.

#### Stage 3: Evaluation module and classic baselines (`stage-3-baselines`)

- **Goal:** The single scoring path, with the first baselines scored through it.
- **Deliverables:**
  - `src/evaluate.py`: reads any `preds_<model>.csv`, checks that it covers every test post, and returns macro-F1, per-class P/R/F1, the confusion matrix and the invalid-output rate. Each is computed for all posts, slang-heavy posts and plain posts. `INVALID` counts as wrong.
  - `src/labels.py`: adds the VADER threshold rule and the FinBERT / CryptoBERT label maps.
  - `notebooks/03_baselines.ipynb`: runs VADER, FinBERT and CryptoBERT, writing `results/preds_vader.csv`, `preds_finbert.csv` and `preds_cryptobert.csv`. It also writes the author's tags as `preds_author_tag.csv`, so author-vs-hand agreement goes through the same scorer. Ends with a small score table from `evaluate.py`.
- **Packages:** `vaderSentiment`, `transformers` + `torch` (FinBERT, CryptoBERT), `matplotlib` (confusion matrices).
- **New concepts:** macro-F1, confusion matrix.
- **Done when:** all four prediction files cover the 300 posts and score without errors, and the CryptoBERT caveat appears next to its score.

#### Stage 4: LLM prompting (`stage-4-llm-prompting`)

- **Goal:** Zero-shot and few-shot baselines from the same small instruct LLM that Stage 5 fine-tunes.
- **Deliverables:**
  - Choose the base model. Default: `Qwen2.5-1.5B-Instruct`, because it needs no access request and trains quickly on a T4. Try 3B only if 1.5B can't follow the one-word format. The choice then stays fixed for Stages 4–7.
  - `src/prompts.py`: the prompt (asks for one word) and the fixed few-shot examples, 2 per class, picked from `train.csv` and approved by Will.
  - `src/labels.py`: adds the reply parser (reply → label or `INVALID`).
  - `notebooks/04_llm_prompting.ipynb`: runs both prompts locally (transformers on MPS, greedy decoding so outputs repeat) and writes `results/preds_llm_zeroshot.csv` and `preds_llm_fewshot.csv`. Raw replies go to `results/replies_<model>.csv` for error analysis.
- **New concepts:** instruct models, zero-shot vs few-shot, greedy decoding.
- **Done when:** both prediction files cover the 300 posts, invalid-output rates are reported, and the scores come from `evaluate.py`.

#### Stage 5: QLoRA fine-tune (`stage-5-lora`)

- **Goal:** A LoRA adapter trained on `train.csv`, plus its test-set predictions.
- **Deliverables:**
  - `notebooks/05_lora_finetune.ipynb`, which runs on a Colab T4:
    - Clones the repo to get `src/` and `data/processed/`, and installs `peft`, `trl` and `bitsandbytes` in Colab only.
    - Loads the base model in 4-bit and trains it to answer the Stage 4 zero-shot prompt with the label word, with the loss computed on the answer only.
    - **Convergence tests before the full run** (required by the professor): (1) overfit about 50 posts, where loss should fall close to zero; (2) a short run on about 1,000 posts, where training and validation loss should both drop and replies should be valid labels. The short run also measures time per step, to estimate the full run.
    - Tracks validation loss, saves checkpoints and the adapter to Google Drive, and predicts on the test set.
  - `results/preds_llm_lora.csv`, downloaded from Colab and committed. The adapter goes in `models.nosync/` locally and is never committed.
- **New concepts:** LoRA, 4-bit quantization, QLoRA, training vs validation loss.
- **Done when:** the notebook runs top to bottom on a fresh Colab session, the prediction file covers the 300 posts and scores through `evaluate.py`, and the training settings and run time are recorded in the notebook.
- **Note:** start with 1 epoch on the 30k set. Drive checkpoints protect against Colab disconnects.

#### Stage 6: Final evaluation and write-up (`stage-6-evaluation`)

- **Goal:** One comparison of every model, and a README that shows the results.
- **Deliverables:**
  - `notebooks/06_evaluation.ipynb`:
    - Scores every `results/preds_*.csv` with `evaluate.py` and writes `results/metrics.csv`.
    - Saves figures to `results/figures/`: macro-F1 by model, split into slang-heavy vs plain, and confusion matrices.
    - Reports author-tag vs hand-label agreement and invalid-output rates.
    - Ends with a short error analysis: a few slang posts the models got wrong.
  - README: results table, one or two figures, caveats, and a complete "Getting started".
- **Done when:** the notebook runs top to bottom, the README numbers match `metrics.csv`, and every caveat from section 1 appears in the README.

#### Stage 7 (optional): Local hosting and latency (`stage-7-local`)

- **Goal:** Run the fine-tuned model on the Mac and measure its speed.
- **Deliverables:**
  - `notebooks/07_local_inference.ipynb`:
    - Loads the base model + adapter with the simplest option that works. First try transformers + `peft` on MPS, the same setup as Stage 4; fall back to GGUF / Ollama only if that fails.
    - Measures latency per post over the 300 test posts (median and 90th percentile).
    - Compares the local predictions with the Colab ones.
  - Latency results added to the README.
- **Done when:** the notebook runs on the Mac and the README reports latency along with the hardware it was measured on.
- **Note:** Colab predicts with a 4-bit base model and the Mac runs a 16-bit one, so a few predictions may differ. Report how many.

---

## 4. Git workflow

Keep it simple:

- **`main` always works.** Never leave it broken.
- **One branch per stage:** `stage-1-data-prep`, `stage-2-test-set`, etc.
- **Small, frequent commits** at logical stopping points, with short imperative messages and a prefix:
  - `feat:` new functionality · `fix:` bug fix · `data:` data/splits · `docs:` README/CLAUDE.md/comments · `refactor:` cleanup, no behavior change · `chore:` setup, dependencies
  - Example: `feat: add VADER baseline predictions`
- **Finish a stage:** make sure notebooks run top to bottom → merge the branch into `main` → push → delete the branch.
- **Never commit:** raw data, model weights or adapters, `.nosync` folders, `.venv`, `.ipynb_checkpoints`, `.DS_Store`, secrets. Set up `.gitignore` in the first stage.
- **Before the first push**, check which GitHub account is configured (`git remote -v`, `git config user.name`, `gh auth status` if available) and confirm with Will. This repo goes on Will's **personal** GitHub. **Never push to `wk715-hub`** (work-only account).
- Don't force-push, rewrite history, or delete branches with unmerged work without asking.

---

## 5. Will presents this project in class

Will has to **present this project and explain the code and the steps**. So:

- Prefer code he can explain line by line over shorter code he can't.
- When you introduce something new, explain what it does and why it's needed before or right after writing it.
- At the end of each stage, give a short recap: what was built, how it works, and what the results mean.
- Flag anything likely to come up as a question in the presentation (design choices, caveats, surprising results).

---

## 6. This repo is public on Will's personal GitHub

It's part of his portfolio, so it must look **clean and polished**:

- Clear, consistent file and folder names; numbered notebooks in run order.
- Every notebook starts with a short markdown header: what it does, inputs, outputs.
- Remove dead code, debug prints, and scratch cells before merging to `main`.
- Keep `README.md` current. As stages finish, fill in **Getting started** (exact setup and run commands) and add results (a metrics table and one or two clear figures).
- Figures: labeled axes, titles, readable fonts, saved to `results/figures/`.
- No secrets, personal paths (e.g., `/Users/...`), or large files in the repo.
