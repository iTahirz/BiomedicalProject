# =============================================================================
# PHASE 3: EVALUATION AND TESTING
# =============================================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# 1. PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_FILE = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results_images')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

print(f"--- STARTING PHASE 3 ---")
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError("Model file missing. Run Phase 2 first.")

# Load Data
data = np.load(DATA_FILE)
# Validation set (Last 3000 samples of training data)
X_val = data['x_train'][-3000:] 
Y_val = data['y_train'][-3000:]
# Test set (Unseen MITDB data)
X_test = data['x_test']

print(f"[Data] Test Set Size (MITDB): {len(X_test)} samples")

# Load Model
model = tf.keras.models.load_model(MODEL_FILE)

# Classes
CLASSES = ['Background', 'P-Wave', 'QRS', 'T-Wave']
COLORS = ['#ecf0f1', '#2ecc71', '#e74c3c', '#3498db']

# -----------------------------------------------------------------------------
# 2. METRICS
# -----------------------------------------------------------------------------
print("\n[Metrics] Computing Confusion Matrix...")
Y_pred_probs = model.predict(X_val, verbose=0)
Y_pred_classes = np.argmax(Y_pred_probs, axis=-1).flatten()
Y_true_classes = Y_val.flatten()

print(classification_report(Y_true_classes, Y_pred_classes, target_names=CLASSES))

# Confusion Matrix Plot
cm = confusion_matrix(Y_true_classes, Y_pred_classes, normalize='true')
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Normalized Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
print("[Saved] confusion_matrix.png")

# -----------------------------------------------------------------------------
# 3. BLIND TEST VISUALIZATION
# -----------------------------------------------------------------------------
print("\n[Visual] Generating Blind Test Plots...")
if len(X_test) > 0:
    indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(indices):
        sample = X_test[idx:idx+1]
        pred = model.predict(sample, verbose=0)
        mask = np.argmax(pred[0], axis=-1)
        
        signal = sample[0].flatten()
        time = np.arange(len(signal)) / 250.0
        
        plt.figure(figsize=(12, 4))
        plt.plot(time, signal, 'k', label='ECG (MITDB)')
        
        for cls in [1, 2, 3]:
            plt.fill_between(time, signal.min(), signal.max(), where=(mask==cls),
                             color=COLORS[cls], alpha=0.4, label=CLASSES[cls])
        
        plt.title(f'Test Sample #{idx}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'test_sample_{i}.png'))
        plt.close()
    print("[Success] Test images saved.")
else:
    print("[Warning] No test data found (X_test is empty). Check Phase 1.")