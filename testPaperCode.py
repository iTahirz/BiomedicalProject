import os
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.signal import butter, filtfilt

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
class Config:
    """
    Configuration parameters derived from the reference paper:
    'ECG signal decomposition using l1 Fourier analysis'.
    """
    DATA_FOLDER = 'testPaper'
    RECORD_NAME = 's0017lre'
    FS = 1000           # Sampling Frequency (Hz)
    N_SAMPLES = 4000    # Window Length (Samples)
    
    # CRITICAL ADJUSTMENT: 
    # We select Channel 9 (Lead V4) instead of Channel 1 (Lead II).
    # Rationale: The amplitude observed in the paper's figures is approx. 2.5mV. 
    # Only precordial leads (like V4) exhibit such high amplitude in this record.
    CHANNEL_IDX = 9     
    
    # ALGORITHM PARAMETERS (Effectively a Low-Pass Filter):
    # M = 32 corresponds to a cutoff frequency of approx. 8 Hz.
    # Formula: f_cut = (M / N) * FS
    M_HARMONICS = 32    
    L1_ITERATIONS = 100 # Iterations for the MM algorithm to converge

# ==========================================
# 2. SIGNAL PROCESSING UTILITIES
# ==========================================
def butter_highpass_filter(data, cutoff, fs, order=5):
    """
    PRE-PROCESSING: Applies a zero-phase Butterworth high-pass filter.
    Purpose: Removes low-frequency baseline wander (drift) to clean the signal
    before the main algorithm.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    # filtfilt applies the filter forward and backward to ensure zero phase distortion
    y = filtfilt(b, a, data)
    return y

def load_ecg_data():
    """
    Loads and pre-processes the specific ECG record.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    record_path = os.path.join(base_dir, Config.DATA_FOLDER, Config.RECORD_NAME)
    
    try:
        # Load the specific channel (Lead V4)
        record = wfdb.rdrecord(record_path, channels=[Config.CHANNEL_IDX], sampto=Config.N_SAMPLES)
        raw_signal = record.p_signal.flatten()
        
        # ADVANCED PRE-PROCESSING:
        # 1. High-Pass Filter (0.5 Hz): Removes baseline wander.
        clean_signal = butter_highpass_filter(raw_signal, cutoff=0.5, fs=Config.FS)
        
        # 2. Centering: Ensures the signal oscillates around zero.
        clean_signal = clean_signal - np.mean(clean_signal)
        
        return clean_signal
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# ==========================================
# 3. MATHEMATICAL CORE (ALGORITHMS)
# ==========================================
def create_fourier_matrix(N, M):
    """
    Constructs the Fourier Basis Matrix Phi (N x M).
    This restricts the reconstruction to the first M low-frequency harmonics.
    """
    n = np.arange(N).reshape(-1, 1)
    k = np.arange(M).reshape(1, -1)
    return np.exp(1j * 2 * np.pi * k * n / N)

def solve_l2_fourier(x, Phi):
    """
    Standard l2-norm minimization (Least Squares).
    Analytic solution: c = (Phi^H * Phi)^-1 * Phi^H * x
    """
    c_l2, _, _, _ = np.linalg.lstsq(Phi, x, rcond=None)
    return c_l2, np.real(np.dot(Phi, c_l2))

def solve_l1_fourier(x, Phi, iterations=100):
    """
    l1-norm minimization using Iterative Reweighted Least Squares 
    (Majorization-Minimization approach).
    This acts as a robust Low-Pass Filter that ignores impulsive outliers (QRS).
    """
    # Initialize with the standard l2 solution
    c_curr, _ = solve_l2_fourier(x, Phi)
    epsilon = 1e-10 # Small constant to prevent division by zero
    
    for r in range(iterations):
        x_rec = np.dot(Phi, c_curr)
        error = x - x_rec
        
        # Calculate weights: Weight is inversely proportional to the absolute error.
        # Large errors (outliers like QRS) get small weights.
        weights = 1.0 / (np.abs(error) + epsilon)
        
        # Solve Weighted Least Squares for the current iteration
        sqrt_W = np.sqrt(weights).reshape(-1, 1)
        Phi_weighted = Phi * sqrt_W
        x_weighted = x.reshape(-1, 1) * sqrt_W
        
        c_new, _, _, _ = np.linalg.lstsq(Phi_weighted, x_weighted, rcond=None)
        c_curr = c_new.flatten()
        
    return c_curr, np.real(np.dot(Phi, c_curr))

# ==========================================
# 4. VISUALIZATION & REPORTING
# ==========================================
def run_experiment():
    # A. Data Loading and Calculation
    ecg_signal = load_ecg_data()
    if ecg_signal is None: return

    Phi = create_fourier_matrix(Config.N_SAMPLES, Config.M_HARMONICS)
    
    # Run Algorithms
    _, rec_l2 = solve_l2_fourier(ecg_signal, Phi)
    _, rec_l1 = solve_l1_fourier(ecg_signal, Phi, iterations=Config.L1_ITERATIONS)
    
    # Calculate Residuals (Fast Component Extraction)
    # Residual = Original Signal - Estimated Slow Component (T-Wave)
    qrs_l2 = ecg_signal - rec_l2
    qrs_l1 = ecg_signal - rec_l1

    # Prepare Axes
    # Note: We use Samples for the X-axis to match the paper's Figure 5 format.
    sample_axis = np.arange(Config.N_SAMPLES) 
    f_cut = (Config.M_HARMONICS / Config.N_SAMPLES) * Config.FS

    # B. Generate Figure
    plt.figure(figsize=(14, 12)) 
    
    # --- SUBPLOT A: T-Wave (Slow Component) ---
    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title(f"A. Slow Component Estimation (T-Wave)", fontsize=12, fontweight='bold', loc='left')
    ax1.plot(sample_axis, ecg_signal, color='#cccccc', label='Raw ECG')
    ax1.plot(sample_axis, rec_l2, color='red', linestyle='--', label='l2 Estimate (Standard)')
    ax1.plot(sample_axis, rec_l1, color='green', linewidth=1.5, label='l1 Estimate (Proposed)')
    
    ax1.set_ylabel("Amplitude\n[mV]", rotation=0, labelpad=40, va='center')
    ax1.set_xlabel("Samples")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- SUBPLOT B: QRS (Residual) ---
    ax2 = plt.subplot(3, 1, 2)
    ax2.set_title("B. Fast Component Extraction (Residual / QRS)", fontsize=12, fontweight='bold', loc='left')
    # Overlay original signal for reference
    ax2.plot(sample_axis, ecg_signal, color='#e0e0e0', label='Original (Ref)', zorder=1)
    ax2.plot(sample_axis, qrs_l2, color='orange', alpha=0.5, label='l2 Residual', zorder=2)
    ax2.plot(sample_axis, qrs_l1, color='blue', linestyle='--', label='l1 Residual', zorder=3)
    
    ax2.set_ylabel("Amplitude\n[mV]", rotation=0, labelpad=40, va='center')
    ax2.set_xlabel("Samples")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- SUBPLOT C: Zoom (Gibbs Phenomenon) ---
    ax3 = plt.subplot(3, 1, 3)
    ax3.set_title("C. Gibbs Phenomenon Detail (Zoom Samples 1600-2200)", fontsize=12, fontweight='bold', loc='left')
    zoom_s, zoom_e = 1600, 2200
    
    ax3.plot(sample_axis[zoom_s:zoom_e], ecg_signal[zoom_s:zoom_e], color='#cccccc', label='Raw ECG')
    ax3.plot(sample_axis[zoom_s:zoom_e], rec_l2[zoom_s:zoom_e], color='red', linestyle='--', label='l2 (Ringing)')
    ax3.plot(sample_axis[zoom_s:zoom_e], rec_l1[zoom_s:zoom_e], color='green', label='l1 (Stable)')
    
    # Highlight the error area
    ax3.fill_between(sample_axis[zoom_s:zoom_e], rec_l2[zoom_s:zoom_e], rec_l1[zoom_s:zoom_e], color='red', alpha=0.15, label='Error Area')

    # Annotations to point out the artifacts
    ax3.annotate('Gibbs Ringing', xy=(1730, 0.16), xytext=(1850, 0.25),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, fontweight='bold')
    
    ax3.set_ylabel("Amplitude\n[mV]", rotation=0, labelpad=40, va='center')
    ax3.set_xlabel("Samples")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # --- INFO BOX ---
    info_text = (
        f"EXPERIMENTAL DATA:\n"
        f"------------------\n"
        f"Dataset: PTB (s0017lre)\n"
        f"Channel: 9 (Lead V4)\n"
        f"Filter: High-Pass 0.5Hz\n"
        f"M (Harmonics): {Config.M_HARMONICS}\n"
        f"Cutoff Freq: {f_cut:.1f} Hz"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=9, fontfamily='monospace',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.5'))

    plt.subplots_adjust(bottom=0.15, hspace=0.4, left=0.15)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_filename = os.path.join(base_dir, "final_exam_report_corrected_channel.png")
    plt.savefig(output_filename, dpi=150)
    print(f"[SUCCESS] Chart saved to: {output_filename}")
    plt.show()

if __name__ == "__main__":
    run_experiment()