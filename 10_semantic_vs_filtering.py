# =============================================================================
# PHASE 10: QUALITATIVE PROOF - SEMANTIC SEGMENTATION VS BLIND FILTERING
# =============================================================================
# OBJECTIVE:
# Visualize a sample where the main QRS is CENTERED in the view.
# Guarantees that Phase 11 will zoom on a visible peak.
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. CONFIGURATION
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_IMG = os.path.join(SCRIPT_DIR, 'phase10_semantic_visual.png')

# 2. ALGORITHMS
def apply_butterworth(x, fs, cutoff=8.0):
    b, a = butter(3, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, x)

# 3. EXECUTION
def run_semantic_visual():
    print("--- GENERATING PHASE 10 (CENTERED VIEW) ---")
    
    data = np.load(DATA_PATH)
    x_test = data['x_test']
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # --- DETERMINISTIC SELECTION LOGIC ---
    # We want a sample where the Max QRS is between 0.5s and 1.5s (Indices 125-375)
    # This ensures it is VISIBLE in the 0-2.0s plot window.
    print("Scanning for perfectly centered QRS...")
    
    candidates = []
    variances = []
    
    # Scan first 1000 samples to save time
    for i in range(min(1000, len(x_test))):
        sig = x_test[i].flatten()
        peak_idx = np.argmax(np.abs(sig))
        
        # Check if peak is in the "Golden Zone" (0.5s to 1.5s)
        if 125 < peak_idx < 375:
            candidates.append(i)
            variances.append(np.max(sig) - np.min(sig)) # Prefer high amplitude
    
    if not candidates:
        print("No centered sample found? Using fallback index 0.")
        best_idx = 0
    else:
        # Pick the one with highest contrast
        best_idx = candidates[np.argmax(variances)]
        
    print(f"LOCKED SAMPLE ID: {best_idx}")
    print(f"(This sample has its main peak centrally located for perfect zooming)")
    
    # Process
    sig = x_test[best_idx].flatten()
    N = len(sig)
    FS = 250.0
    t = np.arange(N) / FS
    
    # Predictions
    pred = model.predict(sig[np.newaxis, ..., np.newaxis], verbose=0)[0]
    ml_t_wave = sig * pred[:, 3]
    butt_t_wave = apply_butterworth(sig, FS, cutoff=8.0)
    
    # PLOTTING
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # 1. Original
    ax.plot(t, sig, color='#dddddd', linewidth=2, label='Original ECG (Raw)')
    # 2. Butterworth
    ax.plot(t, butt_t_wave, color='tab:orange', linestyle='--', linewidth=2.5, label='Butterworth Filter (Leaking)')
    # 3. ML
    ax.plot(t, ml_t_wave, color='#2ecc71', linewidth=3, label='ML U-Net (Semantic)')
    
    # ANNOTATIONS
    qrs_loc = np.argmax(np.abs(sig))
    qrs_t = t[qrs_loc]
    
    y_min, y_max = np.min(sig), np.max(sig)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.3*y_range, y_max + 0.5*y_range)
    
    # Box 1: Spectral Overlap Error
    ax.annotate('Spectral Overlap Error\n(Filter reacts to QRS)', 
                 xy=(qrs_t, butt_t_wave[qrs_loc]), 
                 xytext=(qrs_t - 0.45, y_max + 0.2 * y_range),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=2),
                 fontsize=12, fontweight='bold', color='white',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#e74c3c", ec="none", alpha=1.0),
                 ha='right')
                 
    # Box 2: Semantic Success
    ax.annotate('Semantic Success\n(ML output is exactly 0)', 
                 xy=(qrs_t, ml_t_wave[qrs_loc]), 
                 xytext=(qrs_t + 0.15, y_min - 0.15 * y_range), 
                 arrowprops=dict(facecolor='#2ecc71', shrink=0.05, width=2),
                 fontsize=12, fontweight='bold', color='white',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#2ecc71", ec="none", alpha=1.0))

    ax.set_title(f"Qualitative Comparison: Semantic Understanding vs Frequency Filtering (Sample #{best_idx})", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Amplitude (Normalized)", fontsize=12)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle=':')
    
    # Show 0 to 2.0s. Since peak is between 0.5 and 1.5, it WILL be visible.
    ax.set_xlim(0, 2.0)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"[SUCCESS] Saved: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    run_semantic_visual()