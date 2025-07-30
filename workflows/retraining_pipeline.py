# Copyright 2025 Xuanjiang Liu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# workflows/retraining_pipeline.py
#
# Description:
# This script defines the automated retraining pipeline for the vision
# system. It can be triggered to automatically retrain, register, and
# deploy a new version of the segmentation model.

import os
import sys
import logging
import mlflow
import pathlib

# --- Path Setup ---
# Add the project's root directory to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Setup ---

# Now the import will work correctly
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
        logging.info("Step 1: Kicking off training script...")
        train_model()
        logging.info("✅ Training and registration complete.")

        # --- 2. Get the Latest Model Version ---
        latest_versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["None"])
        if not latest_versions:
            logging.error("❌ No new model version found after training. Aborting.")
            return

        new_model_version = latest_versions[0]
        logging.info(f"Step 2: Found new model version: {new_model_version.version}")

        # --- 3. Promote New Model to "Staging" ---
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
