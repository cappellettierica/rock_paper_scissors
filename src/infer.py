import os, glob, yaml, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from src.data import build_ds, CLASSES   
from src.utils import ensure_dir

def load_cfg(path="configs/default.yaml"):
    with open(path, "r") as f: return yaml.safe_load(f)

def predict_folder(model_path, folder, cfg_path="configs/default.yaml", grid_out="outputs/figures/custom_preds.png"):
    cfg = load_cfg(cfg_path)
    img_size = tuple(cfg["img_size"])
    batch = int(cfg["batch_size"])

    # collect images
    exts = ("*.jpg","*.jpeg","*.png")
    paths = []
    for e in exts: paths += glob.glob(os.path.join(folder, e))
    if not paths:
        print("No images found."); return
    paths = sorted(paths)

    # dummy labels (only need preprocessing)
    y_dummy = np.zeros(len(paths), dtype=np.int32)
    ds = build_ds(paths, y_dummy, img_size, batch, shuffle=False, augment=False)

    model = tf.keras.models.load_model(model_path)
    prob = model.predict(ds, verbose=0)
    pred_idx = np.argmax(prob, axis=1)
    pred_lbl = [CLASSES[i] for i in pred_idx]
    pred_conf = prob[np.arange(len(prob)), pred_idx]

    # grid preview
    cols = 4
    rows = int(np.ceil(len(paths)/cols))
    plt.figure(figsize=(3.5*cols, 3*rows))
    for i, p in enumerate(paths[:rows*cols]):
        img = tf.io.read_file(p)
        img = tf.image.decode_image(img, channels=3)
        plt.subplot(rows, cols, i+1)
        plt.imshow(img.numpy())
        plt.axis("off")
        plt.title(f"{pred_lbl[i]} ({pred_conf[i]:.2f})", fontsize=10)
    ensure_dir(os.path.dirname(grid_out))
    plt.tight_layout()
    plt.savefig(grid_out, dpi=140)
    plt.close()

    # per-file results
    for p, lab, conf in zip(paths, pred_lbl, pred_conf):
        print(f"{os.path.basename(p):<30} -> {lab:>9}  ({conf:.2f})")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    folder = sys.argv[2]
    predict_folder(model_path, folder)

# python -m src.infer outputs/artifacts/medium_cnn.keras "C:/Users/sergi/Documents/imags"