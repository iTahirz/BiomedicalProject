# =============================================================================
# PHASE 8: SYNTHETIC PROOF (REPLICATING PAPER FIGS 1-4)

# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

# --- 1. MATH CORE (L1/L2 ALGORITHMS) ---
def create_fourier_matrix(N, M):
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(M).reshape(1, -1)
    return np.exp(1j * 2 * np.pi * k * n / N)

def solve_l2_fourier(x, Phi):
    c, _, _, _ = np.linalg.lstsq(Phi, x, rcond=None)
    return np.real(np.dot(Phi, c))

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

# --- 2. EXAMPLE 1: SINC FUNCTIONS (Paper Section 3.1, Figs 1-2) ---
def run_sinc_experiment():
    print("[Ex 1] Running Sinc Function Experiment...")
    N = 1000
    n = np.arange(N)
    
    # x1: Fast Sinc (mimicking Fig 1a)
    x1 = np.sinc((n - 500) / 10) 
    # x2: Slow Sinc (mimicking Fig 1b - wider)
    x2 = 0.5 * np.sinc((n - 500) / 50) 
    
    # Mixture
    y = x1 + x2
    
    # Decomposition (trying to extract Slow Component x2)
    M = 20 # Low harmonics count acts as Low-Pass
    Phi = create_fourier_matrix(N, M)
    
    rec_l2 = solve_l2_fourier(y, Phi)
    rec_l1 = solve_l1_fourier(y, Phi, iterations=50)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    plt.suptitle("Example 1: Sinc Functions Separation ", fontsize=14, fontweight='bold')
    
    plt.subplot(3, 1, 1)
    plt.plot(y, 'k', alpha=0.3, label='Mixture (x1 + x2)')
    plt.plot(x2, 'b', linestyle='--', linewidth=2, label='Target Slow Component (x2)')
    plt.title("A. Input Mixture")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(x2, 'b', linestyle='--', alpha=0.5, label='Target')
    plt.plot(rec_l2, 'r', label='L2 Estimate (Standard)')
    plt.title("B. L2 Reconstruction (Note: Gibbs Ringing)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(x2, 'b', linestyle='--', alpha=0.5, label='Target')
    plt.plot(rec_l1, 'g', linewidth=2, label='L1 Estimate (Proposed)')
    plt.title("C. L1 Reconstruction (Better Fit, Less Ringing)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("phase8_ex1_sinc.png")
    print("   -> Saved: phase8_ex1_sinc.png")

# --- 3. EXAMPLE 2: GAUSSIAN KERNELS (Paper Section 3.2, Figs 3-4) ---
def run_gaussian_experiment():
    print("[Ex 2] Running Gaussian Mixture Experiment...")
    N = 1000
    n = np.arange(N)
    
    # Gaussian Parameters (Paper Eq. 10 approximation)
    # x1: Narrow Gaussian (Fast/Impulsive component)
    mu1, b1 = 400, 10
    x1 = np.exp(-((n - mu1)**2) / (2 * b1**2))
    
    # x2: Wide Gaussian (Slow component)
    mu2, b2 = 600, 60
    x2 = 0.8 * np.exp(-((n - mu2)**2) / (2 * b2**2))
    
    y = x1 + x2 # Mixture
    
    # Decomposition
    M = 25 # Cutoff
    Phi = create_fourier_matrix(N, M)
    
    rec_l2 = solve_l2_fourier(y, Phi)
    rec_l1 = solve_l1_fourier(y, Phi, iterations=50)
    
    # Visualization
    plt.figure(figsize=(12, 10))
    plt.suptitle("Example 2: Gaussian Kernels Separation ", fontsize=14, fontweight='bold')
    
    plt.subplot(3, 1, 1)
    plt.plot(y, 'k', alpha=0.3, label='Mixture (x1 + x2)')
    plt.plot(x2, 'b', linestyle='--', linewidth=2, label='Target Slow Component (x2)')
    plt.title("A. Input Mixture (Gaussians)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(x2, 'b', linestyle='--', alpha=0.5, label='Target')
    plt.plot(rec_l2, 'r', label='L2 Estimate')
    plt.text(mu1, 0.5, "Gibbs Artifacts", color='red', fontweight='bold', ha='center')
    plt.title("B. L2 Reconstruction (Fails on Impulsive Noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(x2, 'b', linestyle='--', alpha=0.5, label='Target')
    plt.plot(rec_l1, 'g', linewidth=2, label='L1 Estimate')
    plt.title("C. L1 Reconstruction (Robust to Impulsive Noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("phase8_ex2_gaussian.png")
    print("   -> Saved: phase8_ex2_gaussian.png")

if __name__ == "__main__":
    run_sinc_experiment()
    run_gaussian_experiment()
    print("[SUCCESS] All synthetic proofs generated.")