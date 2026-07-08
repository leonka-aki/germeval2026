# GermEval 2026 - Harmful Content Detection

This repository contains experiments for the GermEval 2026 shared task on harmful content detection in German social media (tweets from a right-wing extremist movement, 2014-2016).

## Subtasks

| Task | Folder | Type | Classes |
|---|---|---|---|
| Call to Action | `data/raw/c2a` | binary | `TRUE` / `FALSE` |
| Attacks on the democratic order | `data/raw/dbo` | multi-class | `nothing`, `criticism`, `agitation`, `subversive` |
| Defamatory offences | `data/raw/def` | binary | `TRUE` / `FALSE` |
| Violence | `data/raw/vio` | multi-class | `nothing`, `propensity`, `call2violence`, `support`, `glorification`, `other` |

All four are heavily class-imbalanced (the harmful classes are rare), so we report macro-F1 rather than accuracy.

## Current results

A single 15%-validation split is noisy for classes with single-digit support, so the numbers below are from stratified cross-validation (mean ± std across folds — see [notebooks/results_summary.ipynb](notebooks/results_summary.ipynb) for the single-split breakdown, confusion matrices, and feature-weight plots):

| Task | TF-IDF + LogReg (5-fold) | gbert-base (3-fold) | gbert-base + class weights (3-fold) |
|---|---|---|---|
| c2a | 0.710 ± 0.005 | 0.815 ± 0.004 | 0.818 ± 0.007 |
| def | 0.711 ± 0.019 | 0.761 ± 0.027 | 0.754 ± 0.037 |
| dbo | 0.524 ± 0.027 | 0.533 ± 0.030 | **0.558 ± 0.021** |
| vio | 0.259 ± 0.025 | 0.324 ± 0.015 | 0.321 ± 0.028 |

Takeaways:
- **gbert-base beats TF-IDF on every subtask**, but CV also revised the story: the earlier single-split comparison made `dbo` look like a clean transformer win (0.44 vs. 0.53); averaged over folds the real gap is much smaller (0.524 vs. 0.533) — that single split was just an unlucky draw for TF-IDF on `dbo`'s rare classes.
- **Class weighting only clearly helps on `dbo`** — better mean *and* lower variance (0.558 ± 0.021 vs. 0.533 ± 0.030). On `c2a`/`def`/`vio` it's a wash or slightly worse, and on `vio` it increases variance rather than reducing it.
- Our read: weighting helps when a class is rare-but-learnable (`dbo`'s `criticism`/`agitation` have hundreds/dozens of training examples), but on `vio`, several classes have so few examples (27-102) that upweighting them just amplifies noise instead of teaching the model anything — a data-scarcity ceiling that reweighting the loss function can't fix. That's consistent with `subversive` (dbo, 53 train examples) and `glorification` (vio, 27 train examples) staying stuck near F1 = 0 across every configuration we've tried.

### Multi-task learning (negative result)

We also tried a single shared gbert-base encoder with one classification head per subtask ([src/training/train_multitask.py](src/training/train_multitask.py)), trained jointly on the union of all four tasks' tweets (23,109 unique tweets by id; 78% carry labels for ≥2 tasks). The idea: since `c2a`/`dbo`/`vio` overlap 72-75% by tweet id and are clearly related phenomena, a shared representation might transfer useful signal between tasks — particularly to help `dbo`/`vio`'s rare classes. It didn't work:

| Task | gbert-base (3-fold CV) | multi-task gbert-base (single split) |
|---|---|---|
| c2a | 0.815 ± 0.004 | 0.789 |
| def | 0.761 ± 0.027 | 0.703 |
| dbo | 0.533 ± 0.030 | 0.519 |
| vio | 0.324 ± 0.015 | 0.300 |

Every task got worse, `c2a`'s drop is outside its own CV noise band (±0.004), and the pattern is consistent across all four — not one unlucky split. Most likely cause: each task's classification head only gets gradient signal from the fraction of the pooled corpus that has that task's label (`c2a`/`dbo`/`vio` ≈ 68-71%, `def` only 14.1%), so within the same 5-epoch budget every task effectively sees *less* of its own labeled data than single-task fine-tuning gives it — and `def`, the lowest-coverage task, took the largest relative hit (0.761 → 0.703). On top of that, the four tasks' losses were summed unweighted, a known naive setup that lets tasks compete for gradient updates rather than reinforce each other.

We're treating this as a real, reportable negative result rather than iterating further (the natural fixes — more epochs, per-task loss weighting, balanced batch sampling — are real additional complexity for uncertain payoff). **The deployable models remain the single-task ones**: `predict.py` never used the multi-task model, so this doesn't affect submissions.

## Approach

1. **Classic, explainable baseline**: TF-IDF + Logistic Regression per subtask. Fast, fully inspectable via feature weights, no GPU required. Good enough to explain in a presentation without black-box concerns.
2. **Transformer comparison**: fine-tune `deepset/gbert-base` per subtask ([src/training/train_transformer.py](src/training/train_transformer.py)) and compare against the baseline, using the same macro-F1 metric and the same train/val split.

Both stages work across all four subtasks via `--task`.

## Setup

Create and activate a Python environment. `torch`/`torchvision` are CUDA-build-specific, so install them first with the wheel index matching your GPU driver (check `nvidia-smi` for your driver's max supported CUDA version), then the rest of the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

If you're on CPU only, or your driver supports a different CUDA version, swap the first `pip install` line for the matching wheel from [download.pytorch.org/whl/torch](https://download.pytorch.org/whl/torch) (or the plain `pip install torch torchvision` for CPU). The TF-IDF baseline doesn't need a GPU at all; the transformer stage does for reasonable training times (CPU fine-tuning of gbert-base on ~15k tweets would take hours instead of minutes).

## Usage

Train and evaluate the TF-IDF + Logistic Regression baseline on one subtask (splits its own train file into train/val, since the shared task's test set has no public labels):

```bash
python -m src.training.train_baseline --task c2a
```

`--task` accepts `c2a`, `dbo`, `def`, or `vio`. Config (n-grams, regularization, val split size, ...) lives in [configs/tfidf_baseline.yaml](configs/tfidf_baseline.yaml). Trained model, metrics, and a classification report are written to `results/tfidf_baseline/<task>/`.

Fine-tune the `gbert-base` transformer on the same task/split for comparison:

```bash
python -m src.training.train_transformer --task c2a
```

Config (epochs, batch size, learning rate, max sequence length, val split size) lives in [configs/gbert.yaml](configs/gbert.yaml) — `data.val_size` and `seed` are kept identical to the TF-IDF config so both models are evaluated on the same held-out split. Needs a GPU for practical training times (see Setup). Trained model, metrics, and a classification report are written to `results/gbert/<task>/`.

### Class-weighting experiment

`dbo`/`vio`'s rarest classes get essentially zero training signal in the plain baseline. `--class-weights` weights gbert-base's loss by inverse class frequency to test whether that helps:

```bash
python -m src.training.train_transformer --task dbo --class-weights
```

Writes to `results/gbert_weighted/<task>/` (kept separate from the unweighted run so both are comparable side by side). TF-IDF's Logistic Regression already uses `class_weight="balanced"` by default, so this flag only applies to the transformer.

### Cross-validation

A single 15%-validation split is noisy for classes with single-digit support (e.g. `subversive` has 8 validation examples). Both training scripts support `--cv N` for N-fold stratified CV instead of the single split — it reports mean ± std macro-F1 across folds and saves `cv_metrics.json`, but does **not** save a deployable model (run again without `--cv` for that):

```bash
python -m src.training.train_baseline --task c2a --cv 5
python -m src.training.train_transformer --task c2a --cv 3 --class-weights
```

CV is cheap for the TF-IDF baseline (seconds) but expensive for the transformer (N full fine-tunes) — budget accordingly.

### Multi-task learning (exploratory, not deployed)

```bash
python -m src.training.train_multitask
```

Trains one shared gbert-base encoder with a head per subtask on the pooled tweet corpus. No `--task` flag — it always trains all four heads at once. See the negative result above before using this; it currently underperforms the single-task models, so `predict.py` doesn't use it.

### Generating a submission

Once a task has a trained model, generate predictions on the real (unlabeled) `*_test_26.csv` in the shared task's required format:

```bash
python -m src.training.predict --team YOURTEAMNAME --run 1 --model gbert
```

`--model` accepts `gbert` (default, the stronger model) or `tfidf`. This writes one CSV per subtask (`id;<label_column>`, semicolon-delimited, `TRUE`/`FALSE` for the binary tasks) to `submissions/<team><run>/`, named `<team><run>_<task>.csv` per the Codabench naming convention. Use `--task c2a` (repeatable) to restrict to specific subtasks. Zip the contents of that folder into `<team><run>.zip` before submitting — this script does not create the zip itself.

Since the shared task allows up to three runs, a natural use of that is one run per model (e.g. `--run 1 --model tfidf`, `--run 2 --model gbert`) to see how each scores on the real leaderboard, not just the local validation split.

The `submissions/` folder doesn't exist until you run this — it's created on demand and gitignored, same as `results/`.