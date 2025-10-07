import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.utils import save_json, ensure_dir, short_ok


def metrics_from_preds(y_true, y_prob):
    y_pred = np.argmax(y_prob, axis=1)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    return acc, prec, rec, f1, y_pred

def save_training_curves(history, outdir, title_prefix=""):
    ensure_dir(outdir)
    # Accuracy
    plt.figure()
    plt.plot(history.history.get("accuracy", []), label="train_acc")
    plt.plot(history.history.get("val_accuracy", []), label="val_acc")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend()
    acc_path = os.path.join(outdir, f"{title_prefix.lower().replace(' ','_')}_accuracy.png")
    plt.savefig(acc_path, bbox_inches="tight"); plt.close()
    # Loss
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
