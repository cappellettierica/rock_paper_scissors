import os, yaml, time, numpy as np, pandas as pd, tensorflow as tf
from src.utils import set_seed, ensure_dir, save_json, short_ok, short_warn
from src.data import discover_images, make_splits, build_ds, class_counts, CLASSES
from src.models import TinyCNN, SmallCNN, MediumCNN, compile_model
from src.eval import metrics_from_preds, standardized_metrics_block, save_training_curves

def make_model(kind, img_size, dropout):
    if kind == "tiny":   return TinyCNN((*img_size,3), dropout=dropout)
    if kind == "small":  return SmallCNN((*img_size,3), dropout=dropout)
    if kind == "medium": return MediumCNN((*img_size,3), dropout=dropout)
    raise ValueError(f"Unknown model kind: {kind}")

def main(cfg_path="configs/default.yaml"):
    with open(cfg_path, "r") as f: cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    data_dir = os.path.expanduser(cfg["data_dir"])
    img_size = tuple(cfg["img_size"])
    batch_size = int(cfg["batch_size"])
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    augment = bool(cfg.get("augment", True))

    out_dir = "outputs"; ensure_dir(out_dir)
    figs_dir = os.path.join(out_dir, "figures"); ensure_dir(figs_dir)
    rep_dir  = os.path.join(out_dir, "reports"); ensure_dir(rep_dir)
    art_dir  = os.path.join(out_dir, "artifacts"); ensure_dir(art_dir)

    # ----- Data discovery
    paths, labels = discover_images(data_dir)
    if len(paths) == 0:
        short_warn("No images found. Check data_dir path.")
        return

    # Basic dataset summary
    dist = class_counts(labels)
    data_summary = {
        "class_distribution": dist,
        "image_shape": [img_size[0], img_size[1], 3],
        "augmentation_applied": bool(augment),
        "other_observations": []
    }
    save_json(data_summary, os.path.join(rep_dir, "data_exploration_summary.json"))
    short_ok("Saved data exploration summary (JSON)")

    # ----- Splits (strict: val for tuning; test final only)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = make_splits(paths, labels, test_size, val_size, seed)

    split_summary = { "train": int(len(X_train)), "validation": int(len(X_val)), "test": int(len(X_test)) }
    save_json(split_summary, os.path.join(rep_dir, "data_splits.json"))
    short_ok("Saved data splits (JSON)")

    # Datasets
    ds_train = build_ds(X_train, y_train, img_size, batch_size, shuffle=True, augment=augment, seed=seed)
    ds_val   = build_ds(X_val,   y_val,   img_size, batch_size, shuffle=False, augment=False, seed=seed)
    ds_test  = build_ds(X_test,  y_test,  img_size, batch_size, shuffle=False, augment=False, seed=seed)

    # Validation check (1-2 lines)
    b = next(iter(ds_train))[0].shape
    short_ok(f"Sample batch shape (train): {b}")

    # ----- Train baseline models
    model_cfgs = cfg["models"]
    metrics_list = []
    comp_rows = []

    def train_one(name, mcfg):
        kind = mcfg["type"]; lr = float(mcfg["lr"]); dp = float(mcfg["dropout"]); epochs = int(mcfg["epochs"])
        model = make_model(kind, img_size, dp)
        compile_model(model, lr)
        t0 = time.time()
        history = model.fit(ds_train, validation_data=ds_val, epochs=epochs, verbose=2)
        train_time = time.time() - t0

        # Save curves
        figs = save_training_curves(history, figs_dir, title_prefix=name)

        # Validation evaluation for selection/tuning only
        yv_prob = model.predict(ds_val, verbose=0)
        _, _, _, _, yv_pred = metrics_from_preds(np.argmax([y for _, y in ds_val.unbatch().batch(1)], axis=2).flatten(),
                                                 yv_prob)
        # Minimal sanity check
        if np.unique(yv_pred).size < 3:
            short_warn(f"{name}: predicted fewer than 3 classes on VAL; consider adjusting dropout/lr.")

        # Final test evaluation
        yt_true = np.argmax([y for _, y in ds_test.unbatch().batch(1)], axis=2).flatten()
        yt_prob = model.predict(ds_test, verbose=0)
        acc, prec, rec, f1, _ = metrics_from_preds(yt_true, yt_prob)

        # Save artifacts
        model_path = os.path.join(art_dir, f"{name}.keras")
        model.save(model_path)
        short_ok(f"Saved model: {model_path}")

        # Structured metrics block (test set only)
        mb = standardized_metrics_block(name, acc, prec, rec, f1, notes="")
        metrics_list.append(mb)

        # Params count
        params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
        comp_rows.append([name, params, mb["accuracy"], mb["precision"], mb["recall"], mb["f1_score"], round(train_time, 2)])

        # Training curves manifest for report
        save_json({"figures": figs}, os.path.join(rep_dir, f"{name}_training_curves.json"))

    # Train: Tiny, Small, Medium
    for name in ["tiny_cnn", "small_cnn", "medium_cnn"]:
        train_one(name, model_cfgs[name])

    # ----- Hyperparameter search on one architecture (val-driven)
    hp = cfg.get("hparam_search", None)
    if hp:
        target = hp["model"]
        base = dict(model_cfgs[target])  # copy
        tried = 0
        best = None
        best_val = -np.inf
        results = []

        # Rebuild datasets (already built) — proceed to search
        for lr in hp["learning_rates"]:
            for dp in hp["dropouts"]:
                for bs in hp["batch_sizes"]:
                    if tried >= hp["max_trials"]: break
                    tried += 1
                    tf.keras.backend.clear_session()
                    model = make_model(base["type"], img_size, dp)
                    compile_model(model, lr)
                    # quick/efficient search: ~40% of base epochs
                    epochs = max(5, int(base["epochs"] * 0.4))
                    hist = model.fit(build_ds(*make_subset(X_train, y_train, frac=1.0), img_size, bs, shuffle=True, augment=True),
                                     validation_data=ds_val, epochs=epochs, verbose=0)
                    val_acc = float(np.max(hist.history.get("val_accuracy", [0.0])))
                    results.append({"lr": lr, "dropout": dp, "batch_size": bs, "val_best_acc": round(val_acc,4)})
                    if val_acc > best_val:
                        best_val = val_acc; best = {"lr": lr, "dropout": dp, "batch_size": bs}
        if best:
            short_ok(f"HP search best (val): {best} with acc={round(best_val,4)}")
            save_json({"search_results": results, "best": best}, os.path.join(rep_dir, "hparam_search_small_cnn.json"))

            # retrain best on train set; test once
            tf.keras.backend.clear_session()
            m = make_model(base["type"], img_size, best["dropout"])
            compile_model(m, best["lr"])
            hist = m.fit(ds_train, validation_data=ds_val, epochs=base["epochs"], verbose=2)
            figs = save_training_curves(hist, figs_dir, title_prefix=f"{target}_hptuned")
            yt_true = np.argmax([y for _, y in ds_test.unbatch().batch(1)], axis=2).flatten()
            yt_prob = m.predict(ds_test, verbose=0)
            acc, prec, rec, f1, _ = metrics_from_preds(yt_true, yt_prob)
            mb = standardized_metrics_block(f"{target}_hptuned", acc, prec, rec, f1, notes=f"Chosen via val acc={round(best_val,4)}")
            metrics_list.append(mb)
            params = int(np.sum([np.prod(v.shape) for v in m.trainable_variables]))
            comp_rows.append([f"{target}_hptuned", params, mb["accuracy"], mb["precision"], mb["recall"], mb["f1_score"], None])
            save_json({"figures": figs}, os.path.join(rep_dir, f"{target}_hptuned_training_curves.json"))

    # ----- Save standardized outputs
    save_json(metrics_list, os.path.join(rep_dir, "metrics.json"))

    # Comparison table (CSV for convenience)
    comp_cols = ["Model Name","Params","Accuracy","Precision","Recall","F1-Score","Training Time"]
    pd.DataFrame(comp_rows, columns=comp_cols).to_csv(os.path.join(rep_dir, "comparison_table.csv"), index=False)

    # Preprocessing steps log
    preprocess_steps = ["image normalization", "image resizing", "data augmentation (train only)", "stratified splitting"]
    save_json(preprocess_steps, os.path.join(rep_dir, "preprocessing_steps.json"))

    short_ok("All artifacts saved in outputs/")

# Utility: optional subset function for fast HP trials (keeps stratification)
def make_subset(X, y, frac=1.0, seed=42):
    import numpy as np
    if frac >= 0.999: return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=int(len(X)*frac), replace=False)
    return X[idx], y[idx]

if __name__ == "__main__":
    main()
