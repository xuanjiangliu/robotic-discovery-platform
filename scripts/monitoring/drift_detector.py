# scripts/monitoring/drift_detector.py
#
# Description:
# This script analyzes the vision service's performance logs to detect
# data drift. It compares a recent window of data to a baseline and
# generates a report with a visualization.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

# --- Configuration ---
LOG_FILE = "logs/vision_service_metrics.csv"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Analysis Parameters
BASELINE_FRAC = 0.5  # Use the first 50% of data as the baseline
DRIFT_THRESHOLD = 0.25 # Drift detected if the recent mean changes by > 25%

def analyze_drift():
    """
    Analyzes the metrics log for data drift and generates a report.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- Starting Drift Analysis ---")

    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        logging.error(f"❌ Log file not found at '{LOG_FILE}'. Please run the vision service first.")
        return

    if len(df) < 50:
        logging.warning("Not enough data to perform drift analysis (need at least 50 records).")
        return

    # --- 1. Split Data into Baseline and Recent ---
    split_index = int(len(df) * BASELINE_FRAC)
    baseline_df = df.iloc[:split_index]
    recent_df = df.iloc[split_index:]

    # For this detector, we focus on the mask coverage
    baseline_mean = baseline_df['mask_coverage_percent'].mean()
    recent_mean = recent_df['mask_coverage_percent'].mean()
    
    percentage_change = ((recent_mean - baseline_mean) / baseline_mean)
    
    logging.info(f"Baseline mask coverage mean: {baseline_mean:.2f}%")
    logging.info(f"Recent mask coverage mean:   {recent_mean:.2f}%")
    logging.info(f"Percentage change: {percentage_change:+.2%}")

    # --- 2. Determine if Drift Occurred ---
    drift_detected = abs(percentage_change) > DRIFT_THRESHOLD

    if drift_detected:
        logging.warning(f"🚨 DRIFT DETECTED! Change ({percentage_change:+.2%}) exceeds threshold of {DRIFT_THRESHOLD:.2%}.")
        print("\nRECOMMENDATION: Trigger the retraining pipeline to adapt the model to the new data distribution.")
        print("-> python workflows/retraining_pipeline.py")
    else:
        logging.info("✅ No significant drift detected.")

    # --- 3. Generate Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df.index, df['mask_coverage_percent'], label='Mask Coverage', color='gray', alpha=0.5, zorder=1)
    ax.plot(df.index, df['mask_coverage_percent'].rolling(window=20).mean(), label='Rolling Mean (20 samples)', color='cornflowerblue', zorder=2)

    # Highlight baseline and recent periods
    ax.axvspan(0, split_index, color='green', alpha=0.1, label=f'Baseline (Mean: {baseline_mean:.2f}%)')
    ax.axvspan(split_index, len(df), color='orange', alpha=0.1, label=f'Recent (Mean: {recent_mean:.2f}%)')

    ax.set_title(f"Drift Analysis: Mask Coverage\nDrift Detected: {drift_detected}", fontsize=16, weight='bold')
    ax.set_xlabel("Log Entry Index")
    ax.set_ylabel("Mask Coverage (%)")
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    report_path = os.path.join(REPORTS_DIR, "drift_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    logging.info(f"✅ Drift analysis report saved to '{report_path}'")
    plt.close()


if __name__ == '__main__':
    analyze_drift()
