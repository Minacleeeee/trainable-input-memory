"""CPU tokenization once per phase, preserving existing masks and padding."""
import torch
from model_support import make_chat_batch


def encode_items(tok, items):
    encoded = []
    for item in items:
        ids, labels = make_chat_batch(tok, [(item['train_q'], item['answer'])],
                                      torch.device('cpu'))
        encoded.append((ids[0], labels[0]))
    return encoded


def collate(encoded, indices, pad_id, device):
    batch = [encoded[i] for i in indices]
    length = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    for i, (row, target) in enumerate(batch):
        ids[i, :len(row)] = row
        labels[i, :len(target)] = target
    return ids.to(device), labels.to(device)
