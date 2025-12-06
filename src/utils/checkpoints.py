import torch
import os

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device='cpu'):
    return torch.load(path, map_location=device)
