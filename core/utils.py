import json, random, os, torch, numpy as np

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
