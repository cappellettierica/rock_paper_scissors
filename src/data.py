import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as K
from src.utils import short_ok, short_warn
import cv2, random
import numpy as np

CLASSES = ["rock", "paper", "scissors"]

def discover_images(root):
    paths, labels = [], []
    exts = ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"]
    for idx, cname in enumerate(CLASSES):
        cpath = os.path.join(root, cname)
        if not os.path.isdir(cpath):
            short_warn(f"Missing class folder: {cpath}")
            continue
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(cpath, e)))
        files = sorted(files)
        paths += files
        labels += [idx] * len(files)
    return np.array(paths), np.array(labels)

def make_splits(paths, labels, test_size=0.15, val_size=0.15, seed=42):
    # 1) hold-out TEST
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    # 2) split remaining into TRAIN/VAL
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_ratio, stratify=y_tmp, random_state=seed
    )
    short_ok(f"Split -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def _basic_aug_layer():
    return K.Sequential([
        K.layers.RandomFlip("horizontal"),
        K.layers.RandomRotation(0.10),
        K.layers.RandomZoom(0.15),
        K.layers.RandomContrast(0.15),
    ])

GREEN_LOWER = np.array([35, 40, 40], dtype=np.uint8)
GREEN_UPPER = np.array([85, 255, 255], dtype=np.uint8)

def remove_green_bg(img_bgr: np.ndarray) -> np.ndarray: # replace with white 
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    out  = img_bgr.copy()
    out[mask != 0] = 255
    return out

def strong_photo_aug(img_bgr: np.ndarray) -> np.ndarray: # flip, rotate, contrast 
    if random.random() < 0.5:
        img_bgr = cv2.flip(img_bgr, 1)
    h, w = img_bgr.shape[:2]
    angle = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    alpha = random.uniform(0.8, 1.3)   # contrast
    beta  = random.randint(-30, 30)    # brightness
    img_bgr = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return img_bgr

def opencv_strong_augment_np(img_rgb: np.ndarray) -> np.ndarray: # combo of green removal and strong aug
    bgr = (np.clip(img_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
    if random.random() < 0.5:  # 50% chance to remove green background
        bgr = remove_green_bg(bgr)
    bgr = strong_photo_aug(bgr)
    rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
    return np.clip(rgb, 0.0, 1.0)

def opencv_strong_augment_tf(img_tensor: tf.Tensor) -> tf.Tensor:
    out = tf.numpy_function(opencv_strong_augment_np, [img_tensor], tf.float32)
    out.set_shape(img_tensor.shape)  # keep (H,W,3)
    return out

def _decode_img(path, img_size):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, img_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def build_ds(paths, labels, img_size=(128,128), batch_size=32,
             shuffle=False, augment=False, seed=42, augment_policy="none"):
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    def _load(path, label):
        x = _decode_img(path, img_size)
        return x, tf.one_hot(label, depth=3)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)

    if augment:
        if augment_policy == "green_strong":
            ds = ds.map(lambda x, y: (opencv_strong_augment_tf(x), y),
                        num_parallel_calls=AUTOTUNE)
        elif augment_policy == "basic":
            aug = _basic_aug_layer()
            ds = ds.map(lambda x, y: (aug(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds

def class_counts(labels):
    from collections import Counter
    c = Counter(labels.tolist())
    return {CLASSES[k]: int(v) for k, v in c.items()}
