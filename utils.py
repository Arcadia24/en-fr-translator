from __future__ import annotations

import builtins
import json
import torch
from torch.utils.data import Dataset

from books import HarryPotter

class Vocab:
    def __init__(self, content: str) -> None:
        self.content = content
        self.chars = sorted(list(set(self.content)))
        self.stoi = {char:i for i, char in enumerate(self.chars)}
        self.itos = {i:char for i, char in enumerate(self.chars)}
        stoi_json = json.dumps(self.stoi, indent=4)
        itos_json = json.dumps(self.itos, indent=4)
        with open("stoi.json", "w") as outfile:
            outfile.write(stoi_json)
        with open("itos.json", "w") as outfile:
            outfile.write(itos_json)

    def __len__(self) -> int:
        return len(self.stoi)

    def __getitem__(self, idx: str | int) -> int | str:
        match type(idx):
            case builtins.str: return self.stoi[idx]
            case builtins.int: return self.itos[idx]

    def encode(self, chars: str) -> list[int]:
        return [self[char] for char in chars]
    
    def decode(self, idxs: list[int]) -> str:
        return ''.join(self[idx] for idx in idxs)
    

class HarryPotterDataset(Dataset):
    def __init__(self, context_length: int, books: HarryPotter, vocab: Vocab, train: bool) -> None:
        super().__init__()
        self.context_length = context_length
        self.train = train
        self.data = torch.tensor(vocab.encode(books.content), dtype=torch.long)
        
        n = int(0.9 * len(self.data))
        self.data = self.data[:n] if train else self.data[n:]

    def __len__(self) -> int:
        return len(self.data) - self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y