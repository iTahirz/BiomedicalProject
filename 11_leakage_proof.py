# =============================================================================
# PHASE 11: THE "SMOKING GUN" - ZOOM LEAKAGE ANALYSIS (SYNCHRONIZED)
# =============================================================================
# OBJECTIVE:
# Zoom in on the EXACT SAME QRS complex shown in Phase 10.
# The peak selection logic is mirrored to ensure temporal alignment.
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 1. SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_IMG = os.path.join(SCRIPT_DIR, 'phase11_leakage_error.png')

# 2. ALGORITHMS
def apply_butterworth(x, fs, cutoff=8.0):
    b, a = butter(3, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, x)

# 3. EXECUTION
def run_leakage_analysis():
    print("--- GENERATING PHASE 11 (ZOOM VIEW) ---")
    
    data = np.load(DATA_PATH)
    x_test = data['x_test']
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # --- SAME DETERMINISTIC SELECTION LOGIC ---
    candidates = []
    variances = []
    
    for i in range(min(1000, len(x_test))):
        sig = x_test[i].flatten()
        peak_idx = np.argmax(np.abs(sig))
        if 125 < peak_idx < 375: # Peak in Golden Zone
            candidates.append(i)
            variances.append(np.max(sig) - np.min(sig))
            
    if not candidates: best_idx = 0
    else: best_idx = candidates[np.argmax(variances)]
        
    print(f"LOCKED SAMPLE ID: {best_idx}")
    
    # Process
    sig = x_test[best_idx].flatten()
    N = len(sig)
    FS = 250.0
    t = np.arange(N) / FS
    
    # Estimates
    pred = model.predict(sig[np.newaxis, ..., np.newaxis], verbose=0)[0]
    ml_t_wave = sig * pred[:, 3]
    butt_t_wave = apply_butterworth(sig, FS, cutoff=8.0)
    
    # ZOOM LOGIC
    # Find the peak (Guaranteed to be the same one as Phase 10)
    qrs_idx = np.argmax(np.abs(sig))
    qrs_t = t[qrs_idx]
    
    # Check: Is this peak within the Phase 10 window (0-2.0s)?
    # Since we filtered for indices 125-375 (0.5s-1.5s), YES IT IS.
    
    # Narrow window +/- 120ms around peak
    window_samples = int(0.12 * FS) 
    start = max(0, qrs_idx - window_samples)
    end = min(N, qrs_idx + window_samples)
    
    t_zoom = t[start:end]
    sig_zoom = sig[start:end]
    butt_zoom = butt_t_wave[start:end]
    ml_zoom = ml_t_wave[start:end]
    
    # PLOTTING
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # 1. Original (Ghost)
    ax.plot(t_zoom, sig_zoom, color='#dddddd', linewidth=5, alpha=0.8, label='Original QRS (Input)')
    
    # 2. Butterworth Error Area
    ax.fill_between(t_zoom, 0, butt_zoom, color='tab:orange', alpha=0.2, label='Butterworth Leakage (Error)')
    ax.plot(t_zoom, butt_zoom, color='tab:orange', linestyle='--', linewidth=3)
    
    # 3. ML Success Line
    ax.plot(t_zoom, ml_zoom, color='#2ecc71', linewidth=4, label='ML Estimate (Correctly Flat)')
    
    # ANNOTATION
    # Position text relative to the error peak
    err_peak_idx = np.argmax(np.abs(butt_zoom))
    err_peak_t = t_zoom[err_peak_idx]
    err_peak_val = butt_zoom[err_peak_idx]
    
    y_max_z = max(np.max(sig_zoom), np.max(butt_zoom))
    y_min_z = min(np.min(sig_zoom), np.min(butt_zoom))
    y_rng_z = y_max_z - y_min_z
    
    ax.set_ylim(y_min_z - 0.2*y_rng_z, y_max_z + 0.4*y_rng_z)
    
    ax.annotate('FALSE PEAK!\n(QRS interpreted as T-wave)', 
                 xy=(err_peak_t, err_peak_val), 
                 xytext=(err_peak_t + 0.03, y_max_z + 0.1*y_rng_z), 
                 arrowprops=dict(facecolor='red', shrink=0.05, width=3),
                 fontsize=14, fontweight='bold', color='white',
                 bbox=dict(boxstyle="round,pad=0.4", fc="#e74c3c", ec="none", alpha=1.0))

    ax.set_title(f"Zoom on QRS Complex: Where Filters Fail (Sample #{best_idx})", fontsize=18, fontweight='bold')
    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Amplitude", fontsize=14)
    ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"[SUCCESS] Saved: {OUTPUT_IMG}")
    plt.show()

if __name__ == "__main__":
    run_leakage_analysis()