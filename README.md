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

TF-IDF + Logistic Regression baseline, validation split (see [notebooks/results_summary.ipynb](notebooks/results_summary.ipynb) for full breakdown, confusion matrices, and feature-weight plots):

| Task | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| c2a | 0.88 | 0.71 | Solid; imperative/modal words are strong, sensible signal |
| def | 0.87 | 0.70 | Solid; top features are literal insults, as expected |
| dbo | 0.84 | 0.44 | Dragged down by `subversive` (53 train examples, F1 = 0) |
| vio | 0.95 | 0.26 | Dragged down by `glorification`/`support`/`other` (< 130 train examples each, F1 ≈ 0) |

Takeaway: the baseline handles the two binary tasks well, but struggles on the rarest classes in the two multi-class tasks — those get single-digit support in the validation split, so a linear bag-of-words model barely sees enough examples to learn from. That's the main thing to watch when the transformer stage is added (either it should be much better here, or the fix needs to be class balancing / more data rather than model choice).

## Approach

1. **Classic, explainable baseline** (current): TF-IDF + Logistic Regression per subtask. Fast, fully inspectable via feature weights, no GPU required. Good enough to explain in a presentation without black-box concerns.
2. **Transformer comparison** (planned): fine-tune `deepset/gbert-base` per subtask and compare against the baseline, using the same macro-F1 metric.

Started with the `c2a` subtask end-to-end; the same scripts work for the other three via `--task`.

## Setup

Create and activate a Python environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Train and evaluate the TF-IDF + Logistic Regression baseline on one subtask (splits its own train file into train/val, since the shared task's test set has no public labels):

```bash
python -m src.training.train_baseline --task c2a
```

`--task` accepts `c2a`, `dbo`, `def`, or `vio`. Config (n-grams, regularization, val split size, ...) lives in [configs/tfidf_baseline.yaml](configs/tfidf_baseline.yaml). Trained model, metrics, and a classification report are written to `results/tfidf_baseline/<task>/`.