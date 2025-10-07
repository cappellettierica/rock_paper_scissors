import json, os, random, numpy as np, tensorflow as tf

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def short_ok(msg):
    print(f"[OK] {msg}")

def short_warn(msg):
    print(f"[WARN] {msg}")
