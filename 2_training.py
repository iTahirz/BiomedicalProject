# =============================================================================
# PHASE 2: MODEL ARCHITECTURE AND SUPERVISED LEARNING
# =============================================================================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

# -----------------------------------------------------------------------------
# 1. ROBUST PATH CONFIGURATION
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')

# Hyperparameters
BATCH_SIZE = 32
EPOCHS_PER_RUN = 10
LEARNING_RATE_INIT = 0.001
LEARNING_RATE_FINE = 0.0001
VALIDATION_SPLIT = 0.2

print(f"[System] Dataset Path: {DATASET_PATH}")
print(f"[System] Model Path:   {MODEL_PATH}")

# -----------------------------------------------------------------------------
# 2. DATA LOADING
# -----------------------------------------------------------------------------
def load_training_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[Error] Dataset missing. Run Phase 1 first.")
    
    print(f"[Data] Loading...")
    data = np.load(filepath)
    X = data['x_train']
    Y_indices = data['y_train']
    Y = tf.keras.utils.to_categorical(Y_indices, num_classes=4)
    print(f"[Data] Loaded {len(X)} samples.")
    return X, Y

# -----------------------------------------------------------------------------
# 3. U-NET 1D MODEL
# -----------------------------------------------------------------------------
def build_unet_1d(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv1D(16, 9, activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv1D(16, 9, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling1D(2)(c1)

    c2 = layers.Conv1D(32, 9, activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Conv1D(32, 9, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling1D(2)(c2)

    c3 = layers.Conv1D(64, 9, activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Conv1D(64, 9, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling1D(2)(c3)

    # Bottleneck
    c4 = layers.Conv1D(128, 9, activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Conv1D(128, 9, activation='relu', padding='same')(c4)

    # Decoder
    u3 = layers.UpSampling1D(2)(c4)
    u3 = layers.Conv1D(64, 2, activation='relu', padding='same')(u3)
    if c3.shape[1] != u3.shape[1]: u3 = layers.ZeroPadding1D((0, c3.shape[1]-u3.shape[1]))(u3)
    m3 = layers.Concatenate()([u3, c3])
    c5 = layers.Conv1D(64, 9, activation='relu', padding='same')(m3)

    u2 = layers.UpSampling1D(2)(c5)
    u2 = layers.Conv1D(32, 2, activation='relu', padding='same')(u2)
    if c2.shape[1] != u2.shape[1]: u2 = layers.ZeroPadding1D((0, c2.shape[1]-u2.shape[1]))(u2)
    m2 = layers.Concatenate()([u2, c2])
    c6 = layers.Conv1D(32, 9, activation='relu', padding='same')(m2)

    u1 = layers.UpSampling1D(2)(c6)
    u1 = layers.Conv1D(16, 2, activation='relu', padding='same')(u1)
    if c1.shape[1] != u1.shape[1]: u1 = layers.ZeroPadding1D((0, c1.shape[1]-u1.shape[1]))(u1)
    m1 = layers.Concatenate()([u1, c1])
    c7 = layers.Conv1D(16, 9, activation='relu', padding='same')(m1)

    outputs = layers.Conv1D(4, 1, activation='softmax')(c7)
    return models.Model(inputs, outputs, name="ECG_UNet_1D")

# -----------------------------------------------------------------------------
# 4. TRAINING EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    X, Y = load_training_data(DATASET_PATH)
    input_shape = (X.shape[1], X.shape[2])
    
    model = None
    if os.path.exists(MODEL_PATH):
        print(f"\n[System] Found existing model. Resuming...")
        try:
            model = models.load_model(MODEL_PATH, compile=False)
            model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_FINE),
                          loss='categorical_crossentropy', metrics=['accuracy'])
        except:
            model = None

    if model is None:
        print("\n[System] Creating new model...")
        model = build_unet_1d(input_shape)
        model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE_INIT),
                      loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks_list = [
        callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
    ]

    model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS_PER_RUN, 
              validation_split=VALIDATION_SPLIT, callbacks=callbacks_list)