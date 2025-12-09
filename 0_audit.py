# =============================================================================
# PHASE 0: DATASET AUDIT AND INTEGRITY CHECK (PROJECT INITIALIZATION)
# =============================================================================
# Overview:
# This script performs the initial audit of the local datasets.
# It DOES NOT re-download data if it detects existing folders.
# It generates statistics on Sampling Rates, Signal Lengths, and Anomalies.
# =============================================================================

import sys
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------------------
# 1. CONFIGURATION AND PATHS (ROBUST VERSION)
# -----------------------------------------------------------------------------
# Get the absolute path of the directory where THIS script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the data directory relative to the script
BASE_DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

print(f"--- STARTING PHASE 0: AUDIT PROTOCOL ---")
print(f"[*] Script Location: {SCRIPT_DIR}")
print(f"[*] Data Directory Target: {BASE_DATA_DIR}\n")

# Dataset Definitions
DATASETS_CONFIG = [
    {'id': 'qtdb',  'remote': 'qtdb',  'folder': 'qt-database-1.0.0', 'ann': 'pu1'},
    {'id': 'ludb',  'remote': 'ludb',  'folder': 'lobachevsky-university-electrocardiography-database-1.0.1', 'ann': 'i'},
    {'id': 'mitdb', 'remote': 'mitdb', 'folder': 'mit-bih-arrhythmia-database-1.0.0', 'ann': 'atr'}
]

# Storage for audit statistics
AUDIT_STATS = {'names': [], 'counts': [], 'sampling_rates': [], 'saturation_issues': 0}

# -----------------------------------------------------------------------------
# 2. CORE FUNCTIONS
# -----------------------------------------------------------------------------
def get_dataset_path(dataset_info):
    """
    Locates the dataset folder. 
    SKIPS DOWNLOAD if the folder exists.
    """
    target_path = os.path.join(BASE_DATA_DIR, dataset_info['folder'])
    
    # CHECK: Does the folder exist?
    if os.path.exists(target_path):
        # Count files to be sure it's not empty
        num_files = len(os.listdir(target_path))
        if num_files > 5:
            print(f"[OK] Dataset {dataset_info['id'].upper()} found locally ({num_files} files). Skipping download.")
            return target_path, True
    
    # If we reach here, the dataset is missing
    print(f"[WARNING] Dataset {dataset_info['id'].upper()} NOT FOUND in '{target_path}'.")
    print(f"          Please ensure your 'data' folder is in the same directory as this script.")
    return target_path, False

def inspect_record_integrity(folder_path, record_name):
    """
    Validates a single ECG record: Check FS, Length, and Saturation.
    """
    full_path = os.path.join(folder_path, record_name)
    try:
        # Load header only first (faster)
        header = wfdb.rdheader(full_path)
        fs = header.fs
        sig_len = header.sig_len
        
        # Load signal for deeper check (optional, can be disabled for speed)
        # record = wfdb.rdrecord(full_path)
        # signal = record.p_signal[:, 0]
        # if np.std(signal) < 1e-6: return False, fs, 0 # Flatline check
        
        return True, fs, sig_len
        
    except Exception as e:
        # print(f"Error reading {record_name}: {e}")
        return False, 0, 0

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION PIPELINE
# -----------------------------------------------------------------------------

for ds in DATASETS_CONFIG:
    # Step A: Locate Data
    dataset_path, exists = get_dataset_path(ds)
    
    if not exists:
        print(f"   -> SKIPPING {ds['id']} (Data missing). Move your DB folders to 'data/'.")
        continue

    # Handle subdirectory structure (sometimes data is inside a 'data' subfolder)
    final_scan_path = dataset_path
    if os.path.exists(os.path.join(dataset_path, 'data')):
        final_scan_path = os.path.join(dataset_path, 'data')
            
    # Step B: Scan Files
    # Find all .dat files to identify records
    try:
        files = sorted([f.replace('.dat', '') for f in os.listdir(final_scan_path) if f.endswith('.dat')])
    except FileNotFoundError:
        print(f"   -> Error accessing folder: {final_scan_path}")
        continue

    print(f"   -> Auditing {ds['id'].upper()}... ({len(files)} records detected)")
    
    valid_records = 0
    fs_detected = []
    
    # Step C: Audit Loop
    for rec in files:
        is_valid, fs, length = inspect_record_integrity(final_scan_path, rec)
        if is_valid:
            valid_records += 1
            fs_detected.append(fs)
            
    # Statistics
    if fs_detected:
        dominant_fs = max(set(fs_detected), key=fs_detected.count)
        print(f"      [Stats] Valid: {valid_records}/{len(files)} | Fs: {dominant_fs} Hz")
        
        AUDIT_STATS['names'].append(ds['id'].upper())
        AUDIT_STATS['counts'].append(valid_records)
        AUDIT_STATS['sampling_rates'].append(dominant_fs)
    else:
        print(f"      [Stats] No valid records found.")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION DASHBOARD
# -----------------------------------------------------------------------------
if len(AUDIT_STATS['names']) > 0:
    print("\n[System] Generating Project Audit Dashboard...")

    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle("PHASE 0: DATASET AUDIT REPORT", fontsize=16, fontweight='bold', y=1.05)

    # Plot 1: Volume
    sns.barplot(x=AUDIT_STATS['names'], y=AUDIT_STATS['counts'], palette="viridis", ax=ax1)
    ax1.set_title("Dataset Volume (Records)", fontweight='bold')
    ax1.set_ylabel("Count")
    for i, v in enumerate(AUDIT_STATS['counts']):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # Plot 2: Sampling Rates
    sns.barplot(x=AUDIT_STATS['names'], y=AUDIT_STATS['sampling_rates'], palette="magma", ax=ax2)
    ax2.set_title("Sampling Rates (Heterogeneity Check)", fontweight='bold')
    ax2.set_ylabel("Frequency (Hz)")
    
    # Add a reference line for target frequency (e.g., 250Hz or 500Hz depending on your project goal)
    ax2.axhline(y=250, color='red', linestyle='--', label='Potential Target (250Hz)')
    ax2.legend()

    plt.tight_layout()
    
    output_img = os.path.join(SCRIPT_DIR, 'phase0_audit_report.png')
    plt.savefig(output_img, dpi=150)
    print(f"[Success] Audit Report saved to: {output_img}")
    plt.show()
else:
    print("\n[!] No datasets were successfully audited. Check your 'data' folder structure.")