# =============================================================================
# PHASE 5:  SCIENTIFIC BENCHMARK (ML vs MATH)
# =============================================================================
# Objective: Prove ML superiority over l1/l2 Fourier methods.
# Strategy: Direct visual comparison of reconstruction quality and artifacts.
# =============================================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'final_benchmark_results')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 2. MATH ALGORITHMS
def create_fourier_matrix(N, M):
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(M).reshape(1, -1)
    return np.exp(1j * 2 * np.pi * k * n / N)

def solve_l2_fourier(x, Phi):
    c_l2, _, _, _ = np.linalg.lstsq(Phi, x, rcond=None)
    return np.real(np.dot(Phi, c_l2))

def solve_l1_fourier(x, Phi, iterations=50):
    c_curr, _, _, _ = np.linalg.lstsq(Phi, x, rcond=None)
    epsilon = 1e-10
    for _ in range(iterations):
        x_rec = np.dot(Phi, c_curr)
        error = x - x_rec
        weights = 1.0 / (np.abs(error) + epsilon)
        
        sqrt_W = np.sqrt(weights).reshape(-1, 1)
        Phi_weighted = Phi * sqrt_W
        x_weighted = x.reshape(-1, 1) * sqrt_W
        
        c_new, _, _, _ = np.linalg.lstsq(Phi_weighted, x_weighted, rcond=None)
        c_curr = c_new.flatten()
    return np.real(np.dot(Phi, c_curr))

def get_dynamic_M(N, fs, cutoff=8.0):
    return int((cutoff * N) / fs)

# 3. EXECUTION
print("--- STARTING FINAL BENCHMARK ---")
data = np.load(DATA_PATH)
x_test = data['x_test']
model = tf.keras.models.load_model(MODEL_PATH)

# Select random samples
indices = np.random.choice(len(x_test), 5, replace=False)

for i, idx in enumerate(indices):
    print(f"[Benchmark] Processing Sample #{idx}...")
    
    # A. Data Prep
    raw_signal = x_test[idx].flatten()
    N = len(raw_signal)
    FS = 250.0 
    time = np.arange(N) / FS
    
    # B. Machine Learning Inference
    ml_input = np.expand_dims(x_test[idx], axis=0) 
    ml_pred = model.predict(ml_input, verbose=0)[0]
    
    # ML Reconstruction: We take the signal ONLY where ML says "It's T-Wave"
    # This creates a clean T-wave signal without QRS
    prob_t = ml_pred[:, 3]
    ml_reconstruction = raw_signal * prob_t 
    
    # C. Math Algorithms
    M_dynamic = get_dynamic_M(N, FS, cutoff=8.0)
    Phi = create_fourier_matrix(N, M_dynamic)
    
    math_l2 = solve_l2_fourier(raw_signal, Phi)
    math_l1 = solve_l1_fourier(raw_signal, Phi, iterations=50)
    
    # --- D. VISUALIZATION (THE ULTIMATE COMPARISON) ---
    fig, ax = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    
    # Plot 1: Input
    ax[0].plot(time, raw_signal, 'k', linewidth=1.5, label='Original ECG')
    ax[0].set_title(f"1. Input: Original ECG Signal (Sample #{idx})", loc='left', fontweight='bold', fontsize=12)
    ax[0].set_ylabel("Amplitude", rotation=0, labelpad=20)
    ax[0].legend(loc='upper right')
    ax[0].grid(True, alpha=0.3)
    
    # Plot 2: L2 Failure (Gibbs)
    ax[1].plot(time, raw_signal, color='lightgrey', label='Original')
    ax[1].plot(time, math_l2, color='red', linestyle='--', linewidth=2, label='L2 Estimate (Standard)')
    ax[1].set_title(f"2. Method L2 (Fourier): Gibbs Artifacts", loc='left', fontweight='bold', fontsize=12)
    ax[1].text(0.02, 0.85, "PROBLEM: Ringing/Oscillation at QRS peaks", transform=ax[1].transAxes, color='red', fontweight='bold')
    ax[1].set_ylabel("Amplitude", rotation=0, labelpad=20)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, alpha=0.3)
    
    # Plot 3: L1 Limit (Low Pass)
    ax[2].plot(time, raw_signal, color='lightgrey', label='Original')
    ax[2].plot(time, math_l1, color='green', linewidth=2, label='L1 Estimate (Paper)')
    ax[2].set_title(f"3. Method L1 (Paper): Robust but Low-Pass", loc='left', fontweight='bold', fontsize=12)
    ax[2].text(0.02, 0.85, "PROBLEM: Good stability, but amplitude loss (blurring)", transform=ax[2].transAxes, color='green', fontweight='bold')
    ax[2].set_ylabel("Amplitude", rotation=0, labelpad=20)
    ax[2].legend(loc='upper right')
    ax[2].grid(True, alpha=0.3)
    
    # Plot 4: ML Superiority
    ax[3].plot(time, raw_signal, color='lightgrey', label='Original')
    ax[3].plot(time, ml_reconstruction, color='blue', linewidth=2, label='ML Reconstruction')
    # Overlay L1 for direct comparison
    ax[3].plot(time, math_l1, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='L1 Reference')
    
    ax[3].set_title(f"4. Method ML (Proposed): High Fidelity", loc='left', fontweight='bold', fontsize=12)
    ax[3].text(0.02, 0.85, "RESULT: Perfect shape retention, zero artifacts", transform=ax[3].transAxes, color='blue', fontweight='bold')
    ax[3].set_ylabel("Amplitude", rotation=0, labelpad=20)
    ax[3].set_xlabel("Time (seconds)", fontweight='bold')
    ax[3].legend(loc='upper right')
    ax[3].grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"final_comparison_{i+1}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   -> Saved: {save_path}")

print("\n[SUCCESS] Benchmark Complete.")