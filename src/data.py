import os, glob, random
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from src.utils import short_ok, short_warn

CLASSES = ["rock", "paper", "scissors"]

def discover_images(root):
    paths, labels = [], []
    for idx, cname in enumerate(CLASSES):
        cpath = os.path.join(root, cname)
        if not os.path.isdir(cpath):
            short_warn(f"Missing class folder: {cpath}")
            continue
        files = sorted(sum([glob.glob(os.path.join(cpath, f"*{ext}")) for ext in [".png", ".jpg", ".jpeg"]], []))
        paths += files
        labels += [idx]*len(files)
    return np.array(paths), np.array(labels)

def make_splits(paths, labels, test_size=0.15, val_size=0.15, seed=42):
    # First split off test
    X_tmp, X_test, y_tmp, y_test = train_test_split(paths, labels, test_size=test_size, stratify=labels, random_state=seed)
    # Then split train/val from remaining
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=seed)
    short_ok(f"Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _decode_img(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def build_ds(paths, labels, img_size=(128,128), batch_size=32, shuffle=False, augment=False, seed=42):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        x = _decode_img(path, img_size)
        return x, tf.one_hot(label, depth=3)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def class_counts(labels):
    from collections import Counter
    c = Counter(labels.tolist())
    return {CLASSES[k]: int(v) for k, v in c.items()}
