# =============================================================================
# PHASE 1: SIGNAL PREPROCESSING, HARMONIZATION, AND SEGMENTATION
# =============================================================================


import wfdb
import numpy as np
import os
import warnings
from scipy.signal import resample, butter, filtfilt

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & ROBUST PATHS
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_FILENAME = os.path.join(SCRIPT_DIR, 'dataset_processed.npz')

# Technical Parameters (Aligned with Project Benchmark)
TARGET_FS = 250              # Target Sampling Rate (Hz)
WINDOW_DURATION = 3          # Window length in seconds
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_DURATION) # 750 samples
SATURATION_THRESHOLD = 5.0   # Maximum allowed saturation percentage (%)

# Dataset Registry
DATASETS_TO_PROCESS = [
    {'id': 'qtdb',  'folder': 'qt-database-1.0.0', 'ann': 'pu1', 'role': 'train'},
    {'id': 'ludb',  'folder': 'lobachevsky-university-electrocardiography-database-1.0.1', 'ann': 'i', 'role': 'train'},
    {'id': 'mitdb', 'folder': 'mit-bih-arrhythmia-database-1.0.0', 'ann': 'atr', 'role': 'test'}
]

# Data Buffers
X_train_buffer, Y_train_buffer = [], []
X_test_buffer = []

# -----------------------------------------------------------------------------
# 2. SIGNAL PROCESSING UTILITIES
# -----------------------------------------------------------------------------
def butter_highpass_filter(data, cutoff, fs, order=5):
    """ Applies a zero-phase Butterworth high-pass filter (0.5 Hz). """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def generate_segmentation_mask(record_path, annotation_ext, signal_length, original_fs):
    """ Parses expert annotations to create a semantic mask. """
    mask = np.zeros(signal_length, dtype=np.int8)
    try:
        ann = wfdb.rdann(record_path, annotation_ext)
        scale_factor = TARGET_FS / original_fs
        samples = (ann.sample * scale_factor).astype(int)
        symbols = np.array(ann.symbol)
        
        for i, (samp, sym) in enumerate(zip(samples, symbols)):
            if samp >= signal_length: break
            if sym == 'p': 
                mask[max(0, samp-12):min(signal_length, samp+13)] = 1
            elif sym in ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V']:
                mask[max(0, samp-15):min(signal_length, samp+15)] = 2
            elif sym == 't':
                mask[max(0, samp-20):min(signal_length, samp+20)] = 3
    except Exception:
        pass
    return mask

def process_dataset_folder(config):
    folder_name = config['folder']
    db_id = config['id']
    role = config['role']
    
    # Robust Path Resolution
    base_path = os.path.join(BASE_DIR, folder_name)
    
    # Handle LUDB nested 'data' folder if present
    if os.path.exists(os.path.join(base_path, 'data')) and os.path.isdir(os.path.join(base_path, 'data')):
        base_path = os.path.join(base_path, 'data')

    if not os.path.exists(base_path):
        print(f"[Warning] Directory for {db_id.upper()} NOT found at: {base_path}")
        return 0

    print(f"Processing {db_id.upper()} (Role: {role.upper()})...")
    
    files = sorted([f.replace('.dat', '') for f in os.listdir(base_path) if f.endswith('.dat')])
    windows_generated = 0
    
    for rec in files:
        try:
            full_path = os.path.join(base_path, rec)
            record = wfdb.rdrecord(full_path)
            signal = record.p_signal[:, 0]
            fs = record.fs
            
            # Quality Control
            min_v, max_v = np.min(signal), np.max(signal)
            sat_pct = (np.sum(signal==min_v) + np.sum(signal==max_v)) / len(signal) * 100
            if sat_pct > SATURATION_THRESHOLD: continue

            # Pre-processing: High-Pass Filter (Match Paper Standard)
            signal = butter_highpass_filter(signal, cutoff=0.5, fs=fs)

            # Harmonization: Resampling
            if fs != TARGET_FS:
                num_samples = int(len(signal) * TARGET_FS / fs)
                signal = resample(signal, num_samples)
            
            # Normalization
            signal = np.nan_to_num(signal)
            if np.std(signal) > 1e-6:
                signal = (signal - np.mean(signal)) / np.std(signal)
            else:
                continue

            # Mask Generation
            mask = None
            if role == 'train':
                mask = generate_segmentation_mask(full_path, config['ann'], len(signal), fs)

            # Windowing
            stride = WINDOW_SAMPLES // 2 if role == 'train' else WINDOW_SAMPLES
            
            for start in range(0, len(signal) - WINDOW_SAMPLES, stride):
                end = start + WINDOW_SAMPLES
                win_x = signal[start:end]
                
                if role == 'train':
                    win_y = mask[start:end]
                    if np.sum(win_y) > 0:
                        X_train_buffer.append(win_x)
                        Y_train_buffer.append(win_y)
                        windows_generated += 1
                else:
                    X_test_buffer.append(win_x)
                    windows_generated += 1
                    
        except Exception:
            continue
            
    print(f"   -> Generated {windows_generated} windows.")
    return windows_generated

# -----------------------------------------------------------------------------
# 3. EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"--- STARTING PHASE 1 ---")
    print(f"[*] Looking for data in: {BASE_DIR}")

    total_wins = 0
    for ds_config in DATASETS_TO_PROCESS:
        total_wins += process_dataset_folder(ds_config)

    print("\n[System] Finalizing arrays...")
    X_train_np = np.array(X_train_buffer)[..., np.newaxis]
    Y_train_np = np.array(Y_train_buffer)[..., np.newaxis]
    X_test_np = np.array(X_test_buffer)[..., np.newaxis]

    np.savez_compressed(OUTPUT_FILENAME, 
                        x_train=X_train_np, 
                        y_train=Y_train_np, 
                        x_test=X_test_np)

    print(f"\n[Success] Phase 1 Complete.")
    print(f"Output: {OUTPUT_FILENAME}")
    print(f"Training Samples: {X_train_np.shape}")
    print(f"Testing Samples:  {X_test_np.shape}")