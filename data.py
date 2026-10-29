
# data.py

from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


DATA_URL = (
    "https://zenodo.org/records/18433203/files/"
    "amylase/by_mutation.csv.gz?download=1"
)

EXPECTED_MD5 = "3acf90685dae45a534c4f12bc9dec6b6"
EXPECTED_COUNTS = (2403, 601, 702)

AA = "ACDEFGHIKLMNPQRSTVWY"

VOCAB = {"<PAD>": 0, "<UNK>": 1}
VOCAB.update({aa: i + 2 for i, aa in enumerate(AA)})


def encode(sequence):
    return [VOCAB[a] for a in sequence]


def download_dataset():
    path = Path("data") / "by_mutation.csv.gz"
    path.parent.mkdir(exist_ok=True)

    if not path.exists():
        print("Downloading FLIP2 from Zenodo...")
        urllib.request.urlretrieve(DATA_URL, path)

    checksum = hashlib.md5(path.read_bytes()).hexdigest()

    if checksum != EXPECTED_MD5:
        raise ValueError(
            f"Dataset checksum mismatch: {checksum}"
        )

    df = pd.read_csv(path)

    assert {"sequence", "target", "set", "validation"} <= set(df.columns)
    assert df["sequence"].notna().all()

    df["sequence"] = df["sequence"].str.upper()
    df["target"] = pd.to_numeric(df["target"], errors="raise")

    assert df["target"].notna().all()
    assert df["sequence"].is_unique
    assert df["sequence"].str.len().nunique() == 1
    assert all(set(s) <= set(AA) for s in df["sequence"])

    return df


def split_dataset(df):
    set_names = df["set"].astype(str).str.lower()
    val = df["validation"].astype(str).str.lower().eq("true")

    assert set(set_names) <= {"train", "validation", "test"}
    assert set(df["validation"].astype(str).str.lower()) <= {"true", "false"}

    train = df[(set_names == "train") & (~val)].copy()
    valid = df[((set_names == "train") & val)
               | (set_names == "validation")].copy()
    test = df[set_names == "test"].copy()

    counts = (len(train), len(valid), len(test))

    if counts != EXPECTED_COUNTS:
        raise ValueError(
            f"Unexpected official split counts: {counts}"
        )

    assert len(train) + len(valid) + len(test) == len(df)

    print(f"Train: {len(train)}")
    print(f"Validation: {len(valid)}")
    print(f"Test: {len(test)}")

    return train, valid, test


def build_reference(train_df):
    sequences = train_df["sequence"].tolist()
    length = len(sequences[0])

    reference = []

    for position in range(length):
        values, counts = np.unique(
            [s[position] for s in sequences],
            return_counts=True
        )
        reference.append(values[np.argmax(counts)])

    return "".join(reference)


class ProteinDataset(Dataset):
    def __init__(self, dataframe, reference, mean, std):
        self.x = torch.tensor(
            [encode(s) for s in dataframe["sequence"]],
            dtype=torch.long
        )

        ref_ids = torch.tensor(
            encode(reference), dtype=torch.long
        )

        self.mutation_mask = (
            self.x != ref_ids.unsqueeze(0)
        ).float()

        targets = dataframe["target"].to_numpy(dtype=np.float32)

        self.y = torch.tensor(
            (targets - mean) / std,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.mutation_mask[idx],
            self.y[idx]
        )


def prepare_data(batch_size=32):
    df = download_dataset()
    train_df, val_df, test_df = split_dataset(df)

    reference = build_reference(train_df)

    mean = float(train_df["target"].mean())
    std = float(train_df["target"].std(ddof=0))
    assert std > 0

    train_ds = ProteinDataset(train_df, reference, mean, std)
    val_ds = ProteinDataset(val_df, reference, mean, std)
    test_ds = ProteinDataset(test_df, reference, mean, std)

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False
    )

    return {
        "train_dataset": train_ds,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "reference": reference,
        "mean": mean,
        "std": std,
        "sequence_length": len(reference),
        "vocab_size": len(VOCAB)
    }
