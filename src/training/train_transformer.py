import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.data.dataset import GermEvalDataset
from src.data.load import load_split
from src.evaluation.metrics import aggregate_cv_metrics, compute_metrics, report
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class WeightedTrainer(Trainer):
    """Trainer variant that weights the cross-entropy loss by inverse class
    frequency, for the class-imbalance experiment (--class-weights)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def encode_labels(labels) -> tuple[dict, dict]:
    """Map each subtask's original labels (bools or class-name strings) to
    contiguous integer ids, sorted for a deterministic, human-readable order."""
    classes = sorted(set(labels), key=str)
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def build_model(config, label2id, id2label, device):
    return AutoModelForSequenceClassification.from_pretrained(
        config["model"]["name"],
        num_labels=len(label2id),
        id2label={i: str(label) for i, label in id2label.items()},
        label2id={str(label): i for label, i in label2id.items()},
    ).to(device)


def run_fold(X_train, y_train, X_val, y_val, config, label2id, id2label, tokenizer, device, class_weights, output_dir):
    model = build_model(config, label2id, id2label, device)
    train_dataset = GermEvalDataset(list(X_train), list(y_train), tokenizer, config["data"]["max_length"])
    val_dataset = GermEvalDataset(list(X_val), list(y_val), tokenizer, config["data"]["max_length"])

    def hf_compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        return compute_metrics(eval_pred.label_ids, predictions)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        fp16=device.type == "cuda",
        report_to=[],
        seed=config["seed"],
    )

    if class_weights is not None:
        trainer = WeightedTrainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            compute_metrics=hf_compute_metrics, class_weights=class_weights,
        )
    else:
        trainer = Trainer(
            model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
            compute_metrics=hf_compute_metrics,
        )
    trainer.train()

    y_pred = np.argmax(trainer.predict(val_dataset).predictions, axis=-1)
    y_val_labels = [id2label[i] for i in y_val]
    y_pred_labels = [id2label[i] for i in y_pred]
    metrics = compute_metrics(y_val_labels, y_pred_labels)
    return metrics, trainer, model, y_val_labels, y_pred_labels


def class_weights_from(y_train, label2id) -> torch.Tensor:
    weights = compute_class_weight("balanced", classes=np.arange(len(label2id)), y=list(y_train))
    return torch.tensor(weights, dtype=torch.float)


def run_cv(df, label2id, id2label, config, tokenizer, device, use_class_weights, n_folds, output_dir):
    label_ids = df["label"].map(label2id)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config["seed"])
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(df["text"], label_ids), start=1):
        y_train = label_ids.iloc[train_idx]
        class_weights = class_weights_from(y_train, label2id) if use_class_weights else None
        metrics, *_ = run_fold(
            df["text"].iloc[train_idx], y_train,
            df["text"].iloc[val_idx], label_ids.iloc[val_idx],
            config, label2id, id2label, tokenizer, device, class_weights, output_dir,
        )
        fold_metrics.append(metrics)
        print(f"[fold {fold}/{n_folds}] macro_f1={metrics['macro_f1']:.3f}")

    summary = aggregate_cv_metrics(fold_metrics)
    print(f"\nmacro_f1: {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f} across {n_folds} folds")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cv_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved CV metrics to {output_dir / 'cv_metrics.json'}")
    print("Run again without --cv to train and save the deployable model used by predict.py.")


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fine-tune gbert-base on a GermEval subtask")
    parser.add_argument("--task", required=True, choices=["c2a", "dbo", "def", "vio"])
    parser.add_argument("--config", default="configs/gbert.yaml")
    parser.add_argument(
        "--class-weights", action="store_true",
        help="Weight the cross-entropy loss by inverse class frequency (class-imbalance experiment)",
    )
    parser.add_argument(
        "--cv", type=int, default=0,
        help="If >0, run N-fold stratified CV for a more robust macro-F1 estimate "
        "(N full fine-tunes - expensive), instead of the normal single train/val run (no model is saved)",
    )
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_split(args.task, "train")
    label2id, id2label = encode_labels(df["label"])

    base_dir_name = config["output"]["directory"] + ("_weighted" if args.class_weights else "")
    output_dir = PROJECT_ROOT / base_dir_name / args.task

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])

    if args.cv:
        print(f"\n=== {args.task} — gbert-base{' (class-weighted)' if args.class_weights else ''}, {args.cv}-fold CV ===")
        run_cv(df, label2id, id2label, config, tokenizer, device, args.class_weights, args.cv, output_dir)
        return

    label_ids = df["label"].map(label2id)
    # same split (seed + ratio) as the TF-IDF baseline, so results are comparable
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"],
        label_ids,
        test_size=config["data"]["val_size"],
        random_state=config["seed"],
        stratify=label_ids,
    )

    class_weights = class_weights_from(y_train, label2id) if args.class_weights else None
    metrics, trainer, model, y_val_labels, y_pred_labels = run_fold(
        X_train, y_train, X_val, y_val, config, label2id, id2label, tokenizer, device, class_weights, output_dir,
    )

    print(f"\n=== {args.task} — gbert-base fine-tuned{' (class-weighted)' if args.class_weights else ''} ===")
    print(json.dumps(metrics, indent=2))
    print("\n" + report(y_val_labels, y_pred_labels))

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "classification_report.txt").write_text(report(y_val_labels, y_pred_labels), encoding="utf-8")
    print(f"\nSaved model and metrics to {output_dir}")


if __name__ == "__main__":
    main()
