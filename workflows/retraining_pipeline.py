# scripts/pipelines/retraining_pipeline.py
#
# Description:
# This script defines the automated retraining pipeline for the EvoFab
# Vision System. It can be triggered by the drift detector to automatically
# retrain, register, and deploy a new version of the segmentation model.
#
# This version uses MLFlow model aliases instead of deprecated stages.
#
# Part of the EvoFab Vision Kit package.

import os
import sys
import logging
import mlflow
import pathlib

# --- Path Setup ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(ROOT_DIR)

# Dynamically import the training script
from scripts.train_segmenter import train_model, MLFLOW_MODEL_NAME, MLRUNS_DIR

# --- Configuration ---
MLFLOW_TRACKING_URI = pathlib.Path(MLRUNS_DIR).as_uri()

# --- Main Pipeline Logic ---
def run_retraining_pipeline():
    """
    Executes the full automated retraining, registration, and deployment pipeline.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("🚀 Starting automated retraining pipeline...")

    # Set the MLFlow tracking URI for this session
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    try:
        # --- 1. Retrain and Register Model ---
        # The train_model function from our script already handles training
        # and registering a new version. We just need to call it.
        logging.info("Step 1: Kicking off training script...")
        train_model()
        logging.info("✅ Training and registration complete.")

        # --- 2. Get the Latest Model Version ---
        # After training, fetch the latest version that was just created.
        latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if not latest_versions:
            logging.error("❌ No new model version found after training. Aborting.")
            return

        new_model_version = latest_versions[0]
        logging.info(f"Step 2: Found new model version: {new_model_version.version}")

        # --- 3. Promote New Model to "Staging" ---
        # Instead of transitioning stages, we now set an alias. This automatically
        # moves the alias if it was pointing to an older version.
        logging.info(f"Step 3: Promoting version {new_model_version.version} by setting 'staging' alias...")
        client.set_registered_model_alias(
            name=MLFLOW_MODEL_NAME,
            alias="staging",
            version=new_model_version.version
        )
        logging.info(f"✅ Successfully pointed 'staging' alias to version {new_model_version.version}.")
        logging.info("🚀 Retraining pipeline finished successfully.")

    except Exception as e:
        logging.error(f"❌ An error occurred during the retraining pipeline: {e}", exc_info=True)

if __name__ == '__main__':
    run_retraining_pipeline()
