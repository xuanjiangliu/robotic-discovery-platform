# Robotic Discovery Platform: Vision System

Welcome to the official repository for the Robotic Discovery Platform System. The **Robotic Discovery Platform** is designed to create a fully integrated system that can independently design, manufacture, and test generations of soft actuators with minimal human intervention. By automating the entire discovery workflow, the platform can explore the design space of soft robotics far more rapidly and efficiently than any manual process, accelerating the pace of innovation.

The current module provides a complete, autonomous MLOps pipeline for training, deploying, and monitoring a real-time computer vision service designed to analyze the physical properties of soft actuators.

## 1. System Architecture

To manage the complexity of this multi-stage process, the platform is built on a **Service-Oriented Architecture (SOA)**. In an SOA, the system is composed of independent, loosely-coupled services that communicate with each other over a network. This design is highly modular, allowing each component (e.g., the vision system, the fabrication system) to be developed, deployed, and scaled independently.

* **gRPC (gRPC Remote Procedure Calls):** All internal communication between services is handled by gRPC. It is a high-performance, open-source RPC framework developed by Google.
    * **Why gRPC?** It uses Protocol Buffers (`.proto` files) to define a strict contract for communication, ensuring that services can exchange data reliably. Its support for bidirectional streaming is essential for applications like the Vision System, which must process a continuous flow of real-time camera data.
* **MLflow:** The entire machine learning lifecycle is managed by MLflow. It is used for tracking experiments, packaging code into reproducible runs, and versioning and deploying models. This provides a robust framework for the MLOps (Machine Learning Operations) aspect of the project.
* **Hardware Interface**: Intel RealSense SDK for 3D camera data.


## 2. Project Structure

The repository is organized into distinct packages and directories, each with a specific responsibility.

```
robotic-discovery-platform/
│
├── logs/
│   └── vision_service_metrics.csv      # Performance logs generated by the server for drift detection.
│
├── ml/                                 # (Local Only - Not tracked by Git)
│   ├── configs/                        # Stores configuration files like camera calibration data.
│   ├── datasets/                       # Stores training and validation image datasets.
│   └── mlruns/                         # The backend store for all MLflow experiments and models.
│
├── pkg/
│   ├── protos/                         # Auto-generated gRPC Python files (vision_pb2.py, vision_pb2_grpc.py).
│   ├── camera.py                       # High-level, thread-safe wrapper for the Intel RealSense camera.
│   ├── geometry_utils.py               # Core functions for 3D point cloud and curvature analysis.
│   └── segmentation_model.py           # PyTorch definition of the U-Net model architecture.
│
├── protos/
│   └── vision.proto                    # The gRPC service and message definitions (the "blueprint" for communication).
│
├── reports/
│   └── drift_report.png                # Visual reports generated by the drift detector script.
│
├── scripts/
│   ├── monitoring/
│   │   └── drift_detector.py           # Analyzes service logs to detect model performance drift.
│   ├── 01_calibrate_camera.py          # Interactive script to perform intrinsic camera calibration.
│   ├── 02_collect_segmentation_data.py # Script to collect and auto-label new training data.
│   └── train_segmenter.py              # Trains the segmentation model and registers it with MLflow.
│
├── services/
│   └── vision_analysis/
│       ├── client.py                   # A gRPC client to visualize the live analysis from the server.
│       └── server.py                   # The core gRPC server that runs the vision analysis service.
│
└── workflows/
    └── retraining_pipeline.py          # Automated pipeline to retrain, register, and deploy a new model version.
```
## 3. Deep Dive: The Vision System
### How It Works:

* **Data Ingestion (`pkg/camera.py`):** The system's entry point is the `Camera` class, which provides a high-level interface to an Intel RealSense 3D camera. To ensure maximum performance and a non-stuttering UI in the client, this module uses a background thread for frame acquisition. This means the main application thread never blocks while waiting for a new frame, allowing it to remain responsive.
* **Real-Time Segmentation (`pkg/segmentation_model.py`):** Each color frame is preprocessed and fed into a deep learning model for semantic segmentation. The model is a **U-Net**, a convolutional neural network architecture defined in `segmentation_model.py`. It has been trained to identify which pixels in the image belong to the actuator, generating a binary "mask" that isolates the object from the background.
* **Geometric Analysis (`pkg/geometry_utils.py`):** The `compute_curvature_profile` function in this module takes the segmentation mask and the corresponding high-fidelity depth data to perform the core analysis:
    * **Point Cloud Generation:** It first generates a dense 3D point cloud of the actuator by projecting the masked depth pixels into 3D space using the camera's intrinsic parameters.
    * **Edge Detection:** It then applies a binning algorithm to robustly find the top edge of the point cloud, which represents the actuator's primary axis of bending.
    * **Spline Fitting & Curvature Calculation:** A mathematical B-spline is fitted to these edge points. The derivatives of this spline are then used to calculate its mean and maximum curvature, providing a quantitative measure of the actuator's deformation.
* **Data Streaming (`services/vision_analysis/server.py`):** The final result—a structured `AnalysisResponse` data object containing the calculated curvature, spline points, and the segmentation mask—is serialized using Protocol Buffers and streamed back to the client over the gRPC connection.

## 4. System Integration & Future Work

The Service-Oriented Architecture is key to the platform's scalability and long-term evolution.

* **Integration via gRPC Contract:** The Vision System operates as a self-contained, black-box service. Other components of the platform, like the Evolutionary Framework, do not need to know the internal details of the vision processing pipeline. They only need to adhere to the gRPC contract defined in `protos/vision.proto`. As long as a client sends an `AnalysisRequest` and is prepared to receive an `AnalysisResponse`, the system will function. This loose coupling makes the entire platform robust and easy to manage.
* **Future Expansion:** This modular design allows for seamless expansion with new capabilities. For example, the 3D printing service and recycling services can be integrated, and as long as they communicate over the same gRPC interface, the rest of the system will not be affected. Because the services are independent, they can be deployed in a distributed or cloud environment (e.g., as separate Docker containers orchestrated by Kubernetes), allowing the platform to scale to handle more complex discovery tasks.

---

## Getting Started: A Step-by-Step Guide

This guide covers the entire workflow, from initial setup to running the autonomous MLOps loop.

### 1. Initial System Setup (One-Time)

These steps only need to be performed once on a new machine.

#### Step 1.1: Prepare the Environment

1.  **Activate Virtual Environment:**
    * On macOS/Linux: `source venv/bin/activate`
    * On Windows: `.\venv\Scripts\activate`
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

#### Step 1.2: Calibrate the Camera

This step calculates the camera's intrinsic parameters, which is essential for accurate 3D measurements.

1.  **Run the Calibration Script:**
    ```bash
    python scripts/01_calibrate_camera.py
    ```
2.  **Perform Calibration:**
    * Hold a checkerboard pattern in front of the camera.
    * Press **`c`** to capture frames from various angles (15-20 captures recommended).
    * Press **`q`** to finish. The script saves the calibration data to `ml/configs/calibration_data.npz`.

### 2. Data Collection & Initial Model Training

This is the most critical phase for ensuring high model performance.

#### Step 2.1: Collect High-Quality Training Data

1.  **Run the Data Collection Script:**
    ```bash
    python scripts/02_collect_segmentation_data.py
    ```
2.  **Data Capture Strategy:**
    * Place an actuator on a flat, non-reflective surface.
    * Press **`s`** to start/stop saving image/mask pairs.
    * Vary the lighting, backgrounds, and actuator poses to create a diverse dataset.
    * Press **`q`** to quit.

#### Step 2.2: Train and Register the First Model

1.  **Run Training:**
    ```bash
    python scripts/train_segmenter.py
    ```
2.  **Start the MLflow UI:**
    ```bash
    mlflow ui
    ```
3.  **Promote the Model in MLflow:**
    * Open your browser to **http://127.0.0.1:5000**.
    * Navigate to the **Models** tab and select the `Actuator-Segmenter` model.
    * Click on the latest version and add the **`staging`** alias. This marks the model as ready for deployment.

### 3. Running the Live System

1.  **Start the Server:** In a terminal, run:
    ```bash
    python services/vision_analysis/server.py
    ```
    The server will automatically load the model with the "staging" alias from MLflow.

2.  **Start the Client:** In a *second* terminal, run:
    ```bash
    python services/vision_analysis/client.py
    ```
    A window will appear showing the live camera feed with real-time segmentation and curvature analysis.

### 4. The Autonomous MLOps Loop

This loop allows you to detect and fix model performance drift automatically.

1.  **Detect Drift:** After running the service for a while, analyze the generated logs:
    ```bash
    python scripts/monitoring/drift_detector.py
    ```
    The script will print its findings and save a visual report to `reports/drift_report.png`.

2.  **Trigger Retraining:** If drift is detected, run the automated retraining pipeline:
    ```bash
    python workflows/retraining_pipeline.py
    ```
    This pipeline retrains the model, registers the new version, and promotes it to "staging", completing the autonomous cycle.

### Appendix: Recompiling Protobuf Files

You only need to do this if you modify the `protos/vision.proto` file.

1.  **Navigate to the project root directory.**
2.  **Run the compiler command:**
    ```bash
    python -m grpc_tools.protoc -I./protos --python_out=./pkg/protos --grpc_python_out=./pkg/protos ./protos/vision.proto
    
