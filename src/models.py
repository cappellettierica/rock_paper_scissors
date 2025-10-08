import tensorflow as tf

def _build_cnn(name, img_size, num_classes, dropout, stages):
    """
    stages: list of (filters, n_convs) tuples.
            MaxPooling2D is applied AFTER each stage except the last one.
    """
    I = tf.keras.Input(img_size)
    x = I
    for i, (f, n) in enumerate(stages):
        for _ in range(n):
            x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
        if i < len(stages) - 1:
            x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    O = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(I, O, name=name)

def TinyCNN(img_size=(128, 128, 3), num_classes=3, dropout=0.2):
    # [Conv 16 x1] -> Pool -> [Conv 32 x1] -> GAP -> Dropout -> Dense
    return _build_cnn("TinyCNN", img_size, num_classes, dropout,
                      stages=[(16, 1), (32, 1)])

def SmallCNN(img_size=(128, 128, 3), num_classes=3, dropout=0.3):
    # [Conv 32 x2] -> Pool -> [Conv 64 x1] -> Pool -> [Conv 128 x1] -> GAP -> Dropout -> Dense
    return _build_cnn("SmallCNN", img_size, num_classes, dropout,
                      stages=[(32, 2), (64, 1), (128, 1)])

def MediumCNN(img_size=(128, 128, 3), num_classes=3, dropout=0.4):
    # [Conv 32 x2] -> Pool -> [Conv 64 x2] -> Pool -> [Conv 128 x2] -> GAP -> Dropout -> Dense
    return _build_cnn("MediumCNN", img_size, num_classes, dropout,
                      stages=[(32, 2), (64, 2), (128, 2)])

def compile_model(model, lr=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
