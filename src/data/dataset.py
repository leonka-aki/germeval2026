import torch
from torch.utils.data import Dataset


class GermEvalDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):

        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }


class MultiTaskGermEvalDataset(Dataset):
    """Like GermEvalDataset, but each tweet carries one label per subtask.
    Tasks the tweet has no annotation for are set to IGNORE_INDEX, which
    nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX) skips automatically."""

    IGNORE_INDEX = -100

    def __init__(self, texts, task_labels: dict, tokenizer, max_length=256):
        self.texts = texts
        self.task_labels = task_labels  # {task: [label_id or IGNORE_INDEX, ...]}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        for task, labels in self.task_labels.items():
            item[f"labels_{task}"] = torch.tensor(labels[idx], dtype=torch.long)
        return item