import os, yaml, time, numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from src.utils import set_seed, ensure_dir, save_json, short_ok
from src.data import discover_images, make_splits, build_ds, class_counts, CLASSES
from src.models import TinyCNN, SmallCNN, MediumCNN, compile_model
from src.eval import (
    metrics_from_preds, standardized_metrics_block, save_training_curves,
    save_class_distribution_bar, save_sample_grid, save_confusion_matrix_fig, save_misclassified_images
)

def make_model(kind, img_size, dropout):
    models = {"tiny": TinyCNN, "small": SmallCNN, "medium": MediumCNN}
    return models[kind]((*img_size, 3), dropout=dropout)

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # config
    set_seed(cfg.get("seed", 42))
    data_dir  = cfg["data_dir"]
    img_size  = tuple(cfg["img_size"])
    batch_sz  = int(cfg["batch_size"])
    test_s    = float(cfg["test_size"])
    val_s     = float(cfg["val_size"])
    model_cfgs = cfg["models"]

    # outputs
    out_dir = "outputs"
    figs_dir, rep_dir, art_dir = [os.path.join(out_dir, p) for p in ["figures","reports","artifacts"]]
    for d in [out_dir, figs_dir, rep_dir, art_dir]: ensure_dir(d)

    # ----- data
    paths, labels = discover_images(data_dir)
    dist = class_counts(labels)
    save_json({"class_distribution": dist, "image_shape": [*img_size,3]}, os.path.join(rep_dir, "data_exploration_summary.json"))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(paths, labels, test_s, val_s, seed=cfg.get("seed",42))
    save_json({"train": len(X_train), "val": len(X_val), "test": len(X_test)}, os.path.join(rep_dir, "data_splits.json"))

    save_class_distribution_bar(dist, CLASSES, os.path.join(figs_dir, "class_distribution.png"))
    save_sample_grid(paths, labels, CLASSES, os.path.join(figs_dir, "sample_grid.png"), n_per_class=6, img_size=img_size)

    # val/test (never augmented)
    ds_val  = build_ds(X_val,  y_val,  img_size, batch_sz, shuffle=False, augment=False).cache().prefetch(tf.data.AUTOTUNE)
    ds_test = build_ds(X_test, y_test, img_size, batch_sz, shuffle=False, augment=False).cache().prefetch(tf.data.AUTOTUNE)

    metrics_list, comp_rows = [], []

    # ----- training helper (per-model augmentation policy)
    def train_one(name, mcfg):
        policy = mcfg.get("augment", "none")
        use_aug = policy != "none"

        ds_train = build_ds(
            X_train, y_train, img_size, batch_sz,
            shuffle=True, augment=use_aug, seed=cfg.get("seed",42), augment_policy=policy
        ).cache().prefetch(tf.data.AUTOTUNE)

        model = make_model(mcfg["type"], img_size, mcfg["dropout"])
        compile_model(model, mcfg["lr"])

        t0 = time.time()
        hist = model.fit(ds_train, validation_data=ds_val,
                         epochs=int(mcfg["epochs"]), verbose=2,
                         callbacks=[EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)])
        train_time = round(time.time() - t0, 2)

        save_training_curves(hist, figs_dir, name)

        yt_true = np.argmax([y for _, y in ds_test.unbatch().batch(1)], axis=2).flatten()
        yt_prob = model.predict(ds_test, verbose=0)
        acc, prec, rec, f1, _ = metrics_from_preds(yt_true, yt_prob)

        save_confusion_matrix_fig(yt_true, np.argmax(yt_prob, 1), CLASSES, os.path.join(figs_dir, f"{name}_cm.png"))
        save_misclassified_images(X_test, yt_true, np.argmax(yt_prob, 1), CLASSES, os.path.join("outputs","misclassified", name))

        model.save(os.path.join(art_dir, f"{name}.keras"))
        mb = standardized_metrics_block(name, acc, prec, rec, f1)
        metrics_list.append(mb)

        params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        comp_rows.append([name, params, acc, prec, rec, f1, train_time])

    for name in ["tiny_cnn", "small_cnn"]:
        train_one(name, model_cfgs[name])

# applying hp tuning to small_cnn to get my medium_cnn
    hp = cfg.get("hparam_search", None) 
    if hp and hp.get("model") == "small_cnn":
        base = dict(model_cfgs["small_cnn"])
        dev_X = np.array(list(X_train) + list(X_val))
        dev_y = np.array(list(y_train) + list(y_val))
        kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=cfg.get("seed",42))

        best, best_cv, results = None, -np.inf, []
        print("[HP] 3-fold CV for small_cnn...")

        for lr in hp["learning_rates"]:
            for dp in hp["dropouts"]:
                for bs in hp["batch_sizes"]:
                    scores = []
                    for tr, va in kf.split(dev_X, dev_y):
                        ds_tr = build_ds(dev_X[tr], dev_y[tr], img_size, int(bs), shuffle=True, augment=False)
                        ds_va = build_ds(dev_X[va], dev_y[va], img_size, int(bs), shuffle=False, augment=False)
                        m = make_model("small", img_size, float(dp)); compile_model(m, float(lr))
                        h = m.fit(ds_tr, validation_data=ds_va, epochs=max(5, int(base["epochs"]*0.4)), verbose=0)
                        scores.append(max(h.history["val_accuracy"]))
                    mean_acc = float(np.mean(scores))
                    results.append({"lr": lr, "dropout": dp, "batch": bs, "val_acc": round(mean_acc,4)})
                    if mean_acc > best_cv:
                        best_cv, best = mean_acc, {"lr": lr, "dropout": dp, "batch": int(bs)}

        save_json({"results": results, "best": best}, os.path.join(rep_dir, "hparam_search_small_cnn_cv.json"))
        short_ok(f"[HP] Best combo: {best} (mean val acc={round(best_cv,4)})")

        # final retrain with SAME strong aug as small_cnn (fair comparison) → save as medium_cnn
        ds_dev = build_ds(dev_X, dev_y, img_size, best["batch"], shuffle=True, augment=True,
                          seed=cfg.get("seed",42), augment_policy="green_strong").cache().prefetch(tf.data.AUTOTUNE)
        m = make_model("small", img_size, float(best["dropout"]))
        compile_model(m, float(best["lr"]))
        hist = m.fit(ds_dev, epochs=int(base["epochs"]), verbose=2)
        save_training_curves(hist, figs_dir, "medium_cnn")

        yt_true = np.argmax([y for _, y in ds_test.unbatch().batch(1)], axis=2).flatten()
        yt_prob = m.predict(ds_test, verbose=0)
        acc, prec, rec, f1, _ = metrics_from_preds(yt_true, yt_prob)

        save_confusion_matrix_fig(yt_true, np.argmax(yt_prob, 1), CLASSES, os.path.join(figs_dir, "medium_cnn_cm.png"))
        save_misclassified_images(X_test, yt_true, np.argmax(yt_prob, 1), CLASSES, os.path.join("outputs","misclassified","medium_cnn"))

        m.save(os.path.join(art_dir, "medium_cnn.keras"))
        mb = standardized_metrics_block("medium_cnn", acc, prec, rec, f1, notes=f"CV-tuned small CNN (best mean val acc={round(best_cv,4)})")
        metrics_list.append(mb)
        params = int(np.sum([np.prod(v.shape) for v in m.trainable_variables]))
        comp_rows.append(["medium_cnn", params, acc, prec, rec, f1, None])

    save_json(metrics_list, os.path.join(rep_dir, "metrics.json"))
    pd.DataFrame(comp_rows, columns=["Model","Params","Acc","Prec","Rec","F1","Time"])\
      .to_csv(os.path.join(rep_dir, "comparison_table.csv"), index=False)
    short_ok("All artifacts saved in outputs/")

if __name__ == "__main__":
    main()
