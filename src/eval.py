import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.utils import save_json, ensure_dir, short_ok
import os, math, shutil, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf 

def metrics_from_preds(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec, f1, y_pred

def save_training_curves(history, outdir, title_prefix=""):
    ensure_dir(outdir)
    # sccuracy
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    acc_path = os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_accuracy.png")
    plt.savefig(acc_path, bbox_inches="tight"); plt.close()
    # loss
    plt.figure()
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    loss_path = os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_loss.png")
    plt.savefig(loss_path, bbox_inches="tight"); plt.close()
    short_ok(f"Saved curves: {acc_path}, {loss_path}")
    return [{"title": f"{title_prefix} Accuracy", "file": os.path.basename(acc_path), "description": "Train/Val accuracy per epoch"},
            {"title": f"{title_prefix} Loss", "file": os.path.basename(loss_path), "description": "Train/Val loss per epoch"}]

def standardized_metrics_block(name, acc, prec, rec, f1, notes=""):
    return {
        "model": name, "accuracy": round(float(acc), 4), "precision": round(float(prec), 4),
        "recall": round(float(rec), 4), "f1_score": round(float(f1), 4), "notes": notes
    }

def save_class_distribution_bar(dist_dict, classes, out_path):
    counts = [dist_dict.get(c, 0) for c in classes]
    plt.figure()
    plt.bar(classes, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140)
    plt.close()

def save_sample_grid(paths, labels, classes, out_path, n_per_class=6, img_size=(128,128)):
    # pick first n_per_class per class (order is fine for a quick grid)
    sel = []
    for cls_idx, cls in enumerate(classes):
        cls_paths = [p for p, y in zip(paths, labels) if int(y) == cls_idx]
        sel.extend(cls_paths[:n_per_class])

    # read images
    imgs = []
    for p in sel:
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3)
        img = tf.image.resize(img, img_size)
        imgs.append(img.numpy().astype(np.uint8))

    cols = n_per_class
    rows = len(classes)
    plt.figure(figsize=(1.8*cols, 1.8*rows))
    k = 0
    for r in range(rows):
        for c in range(cols):
            plt.subplot(rows, cols, r*cols + c + 1)
            plt.imshow(imgs[k])
            plt.axis("off")
            if c == 0:
                plt.ylabel(classes[r], rotation=0, labelpad=20, fontsize=10)
            k += 1
    plt.suptitle("Sample Images per Class", y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_confusion_matrix_fig(y_true, y_pred, classes, out_path, normalize=True):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    if normalize:
        cm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix" + (" (norm.)" if normalize else ""))
    plt.colorbar(fraction=0.046)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # annotate
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            plt.text(j, i, txt, ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()

def save_misclassified_images(x_paths, y_true, y_pred, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for p, t, pr in zip(x_paths, y_true, y_pred):
        if int(t) != int(pr):
            t_name, p_name = classes[int(t)], classes[int(pr)]
            dst_dir = os.path.join(out_dir, f"{t_name}_as_{p_name}")
            os.makedirs(dst_dir, exist_ok=True)
            try:
                shutil.copy2(p, dst_dir)
            except Exception:
                # best-effort copy; skip unreadable items
                pass
            rows.append((p, t_name, p_name))
    # CSV
    csv_path = os.path.join(out_dir, "misclassified_manifest.csv")
    if rows:
        pd.DataFrame(rows, columns=["path","true","pred"]).to_csv(csv_path, index=False)
    return csv_path if rows else None
