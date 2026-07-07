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