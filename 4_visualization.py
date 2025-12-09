# =============================================================================
# PHASE 4: SIGNAL DECOMPOSITION VISUALIZATION (FIXED)
# =============================================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'decomposition_results')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 2. LOADING
if not os.path.exists(DATA_PATH) or not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Data or Model missing.")

print(f"[INFO] Loading resources...")
data_archive = np.load(DATA_PATH)
x_test = data_archive['x_test']
model = tf.keras.models.load_model(MODEL_PATH)

# 3. SAMPLE SELECTION (FIXED)
# We simply take 5 random samples without strict filtering
np.random.seed(42)
total_samples = len(x_test)
if total_samples > 0:
    selected_indices = np.random.choice(total_samples, min(5, total_samples), replace=False)
    print(f"[INFO] Generating plots for samples: {selected_indices}")
else:
    print("[ERROR] Test set is empty!")
    exit()

# 4. PLOTTING ROUTINE
fs = 250.0
time_axis = np.arange(x_test.shape[1]) / fs

for i, idx in enumerate(selected_indices):
    input_sample = np.expand_dims(x_test[idx], axis=0)
    prediction = model.predict(input_sample, verbose=0)[0]
    
    raw_signal = x_test[idx].flatten()
    prob_p = prediction[:, 1]
    prob_qrs = prediction[:, 2]
    prob_t = prediction[:, 3]
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    
    # 1. Original
    axes[0].plot(time_axis, raw_signal, color='black', linewidth=2)
    axes[0].set_title(f'Original ECG Signal (Sample #{idx})', loc='left', fontweight='bold')
    axes[0].set_ylabel('Amplitude\n(Z-Score)', rotation=0, labelpad=60, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. P-Wave
    axes[1].plot(time_axis, prob_p, color='#2ecc71', linewidth=2)
    axes[1].fill_between(time_axis, 0, prob_p, color='#2ecc71', alpha=0.2)
    axes[1].set_title('ML Output: P-Wave Probability', loc='left')
    axes[1].set_ylabel('Prob', rotation=0, labelpad=40, ha='right')
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].grid(True, alpha=0.3)
    
    # 3. QRS
    axes[2].plot(time_axis, prob_qrs, color='#e74c3c', linewidth=2)
    axes[2].fill_between(time_axis, 0, prob_qrs, color='#e74c3c', alpha=0.2)
    axes[2].set_title('ML Output: QRS Complex Probability', loc='left')
    axes[2].set_ylabel('Prob', rotation=0, labelpad=40, ha='right')
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)
    
    # 4. T-Wave
    axes[3].plot(time_axis, prob_t, color='#3498db', linewidth=2)
    axes[3].fill_between(time_axis, 0, prob_t, color='#3498db', alpha=0.2)
    axes[3].set_title('ML Output: T-Wave Probability', loc='left')
    axes[3].set_ylabel('Prob', rotation=0, labelpad=40, ha='right')
    axes[3].set_xlabel('Time (seconds)')
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].grid(True, alpha=0.3)
    
    filename = f"decomposition_result_{i+1}.png"
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {filename}")

print(f"[INFO] Visualization complete.")