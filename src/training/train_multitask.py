import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import default_collate
from transformers import AutoTokenizer, Trainer, TrainingArguments

from src.data.dataset import MultiTaskGermEvalDataset
from src.data.load import TASKS, load_pooled, load_split
from src.evaluation.metrics import compute_metrics, report
from src.models.multitask_gbert import MultiTaskGBert
from src.utils.seed import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def encode_labels(labels) -> tuple[dict, dict]:
    classes = sorted(set(labels), key=str)
    label2id = {label: i for i, label in enumerate(classes)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label


def pooled_val_ids(config) -> set:
    """Union of each task's own held-out validation ids (same seed/ratio as
    the single-task scripts), so evaluation stays comparable to those runs
    and no validation tweet for any task leaks into multi-task training."""
    val_ids = set()
    for task in TASKS:
        df = load_split(task, "train")
        _, val_idx = train_test_split(
            df.index, test_size=config["data"]["val_size"], random_state=config["seed"], stratify=df["label"]
        )
        val_ids |= set(df.loc[val_idx, "id"])
    return val_ids


class MultiTaskTrainer(Trainer):
    def __init__(self, *args, tasks=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tasks = tasks

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = {task: inputs.pop(f"labels_{task}") for task in self.tasks}
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        losses = [
            F.cross_entropy(outputs[task], labels[task], ignore_index=MultiTaskGermEvalDataset.IGNORE_INDEX)
            for task in self.tasks
            if (labels[task] != MultiTaskGermEvalDataset.IGNORE_INDEX).any()
        ]
        loss = torch.stack(losses).sum()
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def evaluate_task(model, tokenizer, texts, task, id2label, device, max_length, batch_size=32):
    model.eval()
    preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoding = tokenizer(batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt").to(device)
        logits = model(input_ids=encoding["input_ids"], attention_mask=encoding["attention_mask"])[task]
        pred_ids = logits.argmax(dim=-1).cpu().tolist()
        preds.extend(id2label[task][i] for i in pred_ids)
    return preds


def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Fine-tune a shared gbert-base encoder with one head per subtask")
    parser.add_argument("--config", default="configs/multitask.yaml")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pooled = load_pooled("train")
    val_ids = pooled_val_ids(config)
    train_df = pooled[~pooled["id"].isin(val_ids)].reset_index(drop=True)
    val_df = pooled[pooled["id"].isin(val_ids)].reset_index(drop=True)
    print(f"pooled: {len(pooled)} tweets total, {len(train_df)} train / {len(val_df)} held out for eval")

    label2id, id2label = {}, {}
    for task in TASKS:
        col = f"label_{task}"
        label2id[task], id2label[task] = encode_labels(pooled[col].dropna())

    task_labels = {}
    for task in TASKS:
        col = f"label_{task}"
        task_labels[task] = [
            label2id[task][v] if pd.notna(v) else MultiTaskGermEvalDataset.IGNORE_INDEX for v in train_df[col]
        ]

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    train_dataset = MultiTaskGermEvalDataset(
        train_df["text"].tolist(), task_labels, tokenizer, config["data"]["max_length"]
    )

    model = MultiTaskGBert(
        config["model"]["name"], num_labels={task: len(label2id[task]) for task in TASKS}
    ).to(device)

    output_dir = PROJECT_ROOT / config["output"]["directory"]

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="epoch",
        fp16=device.type == "cuda",
        report_to=[],
        seed=config["seed"],
        remove_unused_columns=False,  # forward() doesn't take labels_*; Trainer would otherwise strip them
    )
    trainer = MultiTaskTrainer(
        model=model, args=training_args, train_dataset=train_dataset,
        data_collator=default_collate, tasks=list(TASKS),
    )
    trainer.train()

    print("\n=== per-task evaluation on held-out tweets ===")
    for task in TASKS:
        col = f"label_{task}"
        task_val = val_df[val_df[col].notna()]
        y_true = task_val[col].tolist()
        y_pred = evaluate_task(
            model, tokenizer, task_val["text"].tolist(), task,
            id2label, device, config["data"]["max_length"],
        )
        metrics = compute_metrics(y_true, y_pred)
        print(f"\n--- {task} (n={len(task_val)}) ---")
        print(json.dumps(metrics, indent=2))
        print(report(y_true, y_pred))

        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (task_dir / "classification_report.txt").write_text(report(y_true, y_pred), encoding="utf-8")

    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(model_dir)
    task_meta = {
        "model_name": config["model"]["name"],
        "id2label": {task: {str(i): label for i, label in id2label[task].items()} for task in TASKS},
    }
    (model_dir / "task_labels.json").write_text(json.dumps(task_meta, indent=2, default=str), encoding="utf-8")
    print(f"\nSaved shared model to {model_dir} and per-task metrics to {output_dir}/<task>/")


if __name__ == "__main__":
    main()
