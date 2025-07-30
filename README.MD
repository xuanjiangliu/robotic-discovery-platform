EvoFab Vision System
1. Project Overview
This vision system is a production-grade computer vision service designed to provide real-time performance analysis for the EvoFab project. Its primary function is to act as the "sense" component in the autonomous "design-fabricate-test" cycle. It analyzes images of a soft actuator, calculates its curvature, and provides this "validated fitness" score back to the evolutionary algorithm.

The system is built on a Service-Oriented Architecture (SOA) and incorporates a complete MLOps lifecycle to ensure that the underlying computer vision model remains accurate and reliable over time, even as experimental conditions change.

2. System Architecture
The architecture is designed for decoupling, scalability, and maintainability.

VisionAnalysisService: A central gRPC service that encapsulates all computer vision logic. It exposes a single endpoint, AnalyzeStream, which accepts a stream of images and returns a stream of analysis results.

MLFlow Integration: The system is fully integrated with MLFlow for end-to-end MLOps:

Experiment Tracking: All training runs, hyperparameters, and performance metrics are logged.

Model Registry: Trained models are versioned and stored in the registry. The VisionAnalysisService dynamically loads models from the registry, allowing for model updates without service restarts.

Automated Lifecycle: The system includes scripts for automated monitoring and retraining, creating a closed-loop system that can detect and correct for model performance degradation (drift).

3. MLOps Lifecycle
The core of the system's robustness comes from its automated MLOps workflow, which consists of four phases:

Phase 1: Service (gRPC): The core computer vision logic is wrapped in the VisionAnalysisService (server.py), which provides a stable API for any client.

Phase 2: Versioning (MLFlow): The training script (train_segmenter.py) logs all experiments and registers new models in the MLFlow registry. The server dynamically pulls the "Staging" model, decoupling training from deployment.

Phase 3: Monitoring: The server logs key metadata (the size of the detected object) for every prediction it makes. The drift_detector.py script analyzes these logs to detect if the model's behavior is changing over time.

Phase 4: Retraining: If drift is detected, the retraining_pipeline.py can be triggered. It automatically retrains the model on the latest data, registers the new version, and promotes it to "Staging," effectively healing the system.

4. How to Run the System
Follow these steps to set up and run the complete pipeline.

Step 1: Initial Setup
Install Dependencies: Make sure your virtual environment is activated and all required packages are installed.

pip install -r requirements.txt

Calibrate Camera: Run the calibration script and show the checkerboard to the camera.

python scripts/01_calibrate_camera.py

Collect Training Data: Run the data collection script to generate images and masks for training.

python scripts/02_collect_segmentation_data.py

Step 2: Train the First Model
Rename Training Script: Make sure scripts/03_train_segmenter.py has been renamed to scripts/train_segmenter.py.

Run Training: This script will train the model and register the first version in the MLFlow registry.

python scripts/train_segmenter.py

Promote Model (First time only): Open the MLFlow UI (mlflow ui) and manually transition the newly trained model version from "None" to the "Staging" stage. This is only necessary for the very first model.

Step 3: Run the Service
Start the Server: The server will automatically load the "Staging" model from the registry.

python services/vision_analysis/server.py

Start the Client: In a separate terminal, run the client to connect to the server and see the live analysis.

python services/vision_analysis/client.py

As the server runs, it will populate logs/vision_service_log.csv with performance data.

Step 4: The Autonomous Loop (Simulating Drift and Retraining)
Check for Drift: Run the drift detector to analyze the logs.

python scripts/monitoring/drift_detector.py

This will generate a report in reports/drift_analysis_plot.png.

Trigger Retraining: If drift is detected, run the retraining pipeline.

python scripts/pipelines/retraining_pipeline.py

This will automatically train a new model, register it, and promote it to "Staging". The next time the VisionAnalysisService needs a model, it will automatically use this new, improved version.

5. Key Files
services/vision_analysis/server.py: The core gRPC service.

services/vision_analysis/client.py: A test client for the gRPC service.

scripts/train_segmenter.py: Trains the model and registers it with MLFlow.

scripts/monitoring/drift_detector.py: Analyzes server logs to detect performance degradation.

scripts/pipelines/retraining_pipeline.py: The automated workflow to retrain and deploy a new model.

mlruns/: The directory where all MLFlow experiments, artifacts, and registered models are stored.