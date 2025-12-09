# =======================================================
# PHASE 6: SEMANTIC PROOF (P-WAVE vs T-WAVE SEPARATION)


import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. SETUP
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'ecg_unet_model.keras')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'semantic_proof_results')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

# 2. MATH ALGORITHM (L1 ONLY)
def create_fourier_matrix(N, M):
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(M).reshape(1, -1)
    return np.exp(1j * 2 * np.pi * k * n / N)

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
print("--- STARTING SEMANTIC PROOF GENERATION ---")
data = np.load(DATA_PATH)
x_test = data['x_test']
model = tf.keras.models.load_model(MODEL_PATH)

# We define a helper to find a sample with distinct P-waves for clear demonstration
# We check a few random samples until we find one with good variance
indices = np.random.choice(len(x_test), 20, replace=False)

count = 0
for idx in indices:
    if count >= 3: break # Generate 3 good examples
    
    # A. Prepare Data
    raw_signal = x_test[idx].flatten()
    N = len(raw_signal)
    FS = 250.0 
    time = np.arange(N) / FS
    
    # B. Run ML Inference
    ml_input = np.expand_dims(x_test[idx], axis=0) 
    ml_pred = model.predict(ml_input, verbose=0)[0]
    
    # Extract Probabilities
    prob_p = ml_pred[:, 1]   # Class 1: P-Wave
    prob_t = ml_pred[:, 3]   # Class 3: T-Wave
    
    # IF the sample has no clear P-waves (low probability), skip it to find a better example
    if np.max(prob_p) < 0.5: continue
    
    count += 1
    print(f"[Proof] Analyzing Sample #{idx} (Good P-waves detected)...")

    # C. Run Math L1
    M_dynamic = get_dynamic_M(N, FS, cutoff=8.0)
    Phi = create_fourier_matrix(N, M_dynamic)
    math_l1 = solve_l1_fourier(raw_signal, Phi, iterations=50)
    
    # D. Visualization: The Separation Proof
    fig, ax = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    # 1. Original
    ax[0].plot(time, raw_signal, 'k', linewidth=1.5)
    ax[0].set_title(f"1. Input: Original ECG (Sample #{idx})", loc='left', fontweight='bold')
    ax[0].set_ylabel("Amplitude", rotation=0, labelpad=20)
    ax[0].grid(True, alpha=0.3)
    
    # 2. Math L1 (The "Blob")
    ax[1].plot(time, raw_signal, color='lightgrey', alpha=0.5)
    ax[1].plot(time, math_l1, color='green', linewidth=2, label='L1 Estimation')
    ax[1].set_title(f"2. Math Approach (L1 Fourier): Indistinguishable P & T", loc='left', fontweight='bold')
    ax[1].text(0.02, 0.85, "LIMITATION: P-wave and T-wave are merged into one signal", transform=ax[1].transAxes, color='green', fontweight='bold')
    ax[1].set_ylabel("Low Freq\nComponent", rotation=0, labelpad=20)
    ax[1].legend(loc='upper right')
    ax[1].grid(True, alpha=0.3)
    
    # 3. ML P-Wave ONLY
    p_wave_signal = raw_signal * prob_p # Extract amplitude based on probability
    ax[2].plot(time, raw_signal, color='lightgrey', alpha=0.3)
    ax[2].plot(time, p_wave_signal, color='#2ecc71', linewidth=2, label='ML P-Wave')
    ax[2].fill_between(time, 0, p_wave_signal, color='#2ecc71', alpha=0.2)
    
    ax[2].set_title(f"3. ML Approach: Isolated P-Wave", loc='left', fontweight='bold')
    ax[2].set_ylabel("Extracted\nP-Wave", rotation=0, labelpad=20)
    ax[2].legend(loc='upper right')
    ax[2].grid(True, alpha=0.3)
    
    # 4. ML T-Wave ONLY
    t_wave_signal = raw_signal * prob_t
    ax[3].plot(time, raw_signal, color='lightgrey', alpha=0.3)
    ax[3].plot(time, t_wave_signal, color='#3498db', linewidth=2, label='ML T-Wave')
    ax[3].fill_between(time, 0, t_wave_signal, color='#3498db', alpha=0.2)
    
    ax[3].set_title(f"4. ML Approach: Isolated T-Wave", loc='left', fontweight='bold')
    ax[3].set_ylabel("Extracted\nT-Wave", rotation=0, labelpad=20)
    ax[3].set_xlabel("Time (seconds)", fontweight='bold')
    ax[3].legend(loc='upper right')
    ax[3].grid(True, alpha=0.3)
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, f"semantic_proof_{count}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"   -> Generated Proof: {save_path}")

print("\n[SUCCESS] Semantic Proof Generation Complete.")