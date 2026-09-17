# L-DriX-Net: Lightweight Driver eXplainable Network

An end-to-end multi-modal deep learning and computer vision framework for **driver gaze estimation**, **driver attention heatmap prediction**, and **temporal driver state & distraction monitoring**.

L-DriX-Net integrates lightweight facial landmark geometry, head-pose estimation, 24-D gaze representation, cross-modal attention fusion, and a modern React-based web dashboard.

---

## Table of Contents

- [Overview & Architecture](#overview--architecture)
- [Project Directory Structure](#project-directory-structure)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
- [Installation & Setup](#installation--setup)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Python Virtual Environment](#2-python-virtual-environment)
  - [3. Install Python Dependencies](#3-install-python-dependencies)
  - [4. Install Frontend Dependencies](#4-install-frontend-dependencies)
  - [5. Model Weights & Data Setup](#5-model-weights--data-setup)
- [How to Run the Project](#how-to-run-the-project)
  - [Method 1: Run the Full-Stack Web Application (Recommended)](#method-1-run-the-full-stack-web-application-recommended)
  - [Method 2: Standalone Pipeline & Inference Scripts](#method-2-standalone-pipeline--inference-scripts)
  - [Method 3: Model Training & Evaluation](#method-3-model-training--evaluation)
  - [Method 4: Component Verification Tests](#method-4-component-verification-tests)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Overview & Architecture

L-DriX-Net combines computer vision geometric detection with deep neural networks:

```
                  ┌────────────────────────────────────────┐
                  │          Driver Face Image             │
                  └──────┬──────────────────────────┬──────┘
                         │                          │
                         ▼                          ▼
               ┌──────────────────┐       ┌──────────────────┐
               │    LFEM Face     │       │ YuNet Detector & │
               │     Encoder      │       │ Geometry Pipeline│
               └─────────┬────────┘       └─────────┬────────┘
                         │                          │
                         │                   24-D Gaze Vector
                         │                          │
┌───────────────┐        │                          ▼
│  Scene Image  │        │                ┌──────────────────┐
└───────┬───────┘        │                │   Gaze Encoder   │
        │                │                └─────────┬────────┘
        ▼                │                          │
┌───────────────┐        │                          │
│ Scene Encoder │        │                          │
└───────┬───────┘        │                          │
        │                │                          │
        └───────┬────────┘                          │
                ▼                                   │
      ┌───────────────────┐                         │
      │ ICFM Cross Fusion │◄────────────────────────┘
      └─────────┬─────────┘
                ▼
      ┌───────────────────┐
      │Temporal Attention │
      └─────────┬─────────┘
                ├──────────────────────────┐
                ▼                          ▼
      ┌───────────────────┐      ┌───────────────────┐
      │  Prediction Head  │      │ Spatial Decoder   │
      │ (Gaze Coordinates │      │(Attention Heatmap)│
      │   & Driver State) │      └───────────────────┘
      └───────────────────┘
```

1. **Lightweight Facial Encoder (LFEM):** Extracts high-level spatial facial cues.
2. **Scene Encoder:** Extracts road and environment features.
3. **YuNet & Gaze Geometry Pipeline:** Detects 5 facial landmarks, extracts eye geometry, estimates head pose (Yaw, Pitch, Roll), and constructs a normalized 24-D gaze vector.
4. **Iterative Cross-Fusion Module (ICFM):** Adaptively fuses driver features with the driving scene context.
5. **Temporal Attention & Spatial Decoder:** Generates explainable 2D attention heatmaps projected over the driving scene.
6. **Prediction Head:** Predicts calibrated gaze coordinates and driver state classifications (Attentive, Drowsy, Distracted).

---

## Project Directory Structure

```text
L-DriX-Net/
├── backend/
│   ├── detection/             # Face detection, eye geometry, head pose & gaze estimation
│   ├── models/                # Pretrained YuNet ONNX face detector
│   ├── inference.py           # LDriXNet inference engine
│   ├── main.py                # FastAPI REST API server
│   ├── test_inference.py      # Backend inference test
│   └── check_metadata.py      # Metadata check utility
├── checkpoints/               # Trained PyTorch model weights (best_model.pth)
├── configs/                   # Configuration files
├── dataset/                   # Dataset folder (Subject01_1_data, etc.)
├── frontend/                  # React + Vite web dashboard
│   ├── src/                   # React components, pages, and services
│   ├── package.json           # Node dependencies
│   └── vite.config.js         # Vite bundler configuration
├── models/                    # L-DriX-Net PyTorch architecture modules
│   ├── ldrixnet.py            # Primary L-DriX-Net network
│   ├── lfem.py                # Lightweight Face Encoder
│   ├── scene_encoder.py       # Scene Encoder
│   ├── gaze_encoder.py        # Gaze Encoder
│   ├── icfm.py                # Iterative Cross-Fusion Module
│   ├── temporal_attention.py  # Temporal Attention Module
│   ├── spatial_projection.py  # Spatial Projection Module
│   ├── decoder.py             # Attention Heatmap Decoder
│   └── heads.py               # Output Prediction Heads
├── scripts/                   # Dataset preparation, frame extraction, heatmaps
├── utils/                     # Dataloaders, losses, and helper functions
├── xai/                       # Explainability (Grad-CAM, saliency)
├── config.py                  # Global training and inference hyperparameters
├── predict.py                 # Standalone prediction script
├── train.py                   # Model training script
├── validate.py                # Model evaluation and metrics script
├── requirements.txt           # Python package requirements
└── .gitignore                 # Git ignore rules
```

---

## Prerequisites & System Requirements

### Hardware Requirements
- **CPU:** Intel Core i5 / AMD Ryzen 5 or higher.
- **RAM:** Minimum 8 GB (16 GB recommended).
- **GPU (Optional but Recommended):** NVIDIA GPU with CUDA 11.8 or 12.x for accelerated training and real-time inference. Runs seamlessly on CPU as fallback.
- **Camera:** Standard USB webcam or built-in camera for live driver monitoring.

### Software Requirements
- **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS.
- **Python:** Python 3.10 to 3.14 (64-bit).
- **Node.js:** Node.js v18.0.0 or higher (v20+ recommended) with `npm`.
- **Git:** Installed on your path.

---

## Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/L-DriX-Net.git
cd L-DriX-Net
```

### 2. Python Virtual Environment

Create and activate an isolated virtual environment:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

Upgrade `pip` and install the required dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **GPU Acceleration with CUDA (Optional):**  
> If you have an NVIDIA GPU, install the CUDA-enabled build of PyTorch:
> ```bash
> # For CUDA 12.1:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> # For CUDA 11.8:
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

### 4. Install Frontend Dependencies

Navigate into the `frontend` directory and install the Node packages:

```bash
cd frontend
npm install
cd ..
```

### 5. Model Weights & Data Setup

1. **Face Detector (Pretrained YuNet):**  
   The lightweight YuNet ONNX face detection model is already bundled at:  
   `backend/models/face_detection_yunet_2026may.onnx`
2. **Trained Checkpoint (`best_model.pth`):**  
   Ensure your trained weights are present in the `checkpoints/` directory:
   ```text
   checkpoints/best_model.pth
   ```
   *(If training from scratch, `train.py` will generate this file automatically).*
3. **Dataset (For Training/Offline Evaluation):**  
   Place dataset folders under `dataset/` (e.g. `dataset/Subject01_1_data`), which should contain:
   - `face_ims/`
   - `scene_ims/`
   - `heatmaps/`
   - Gaze label coordinates / files

---

## How to Run the Project

### Method 1: Run the Full-Stack Web Application (Recommended)

The full-stack application consists of a FastAPI backend service and a Vite + React frontend dashboard.

#### Step 1: Start Backend API Server
Open a terminal in the project root with the virtual environment activated:

```powershell
# Windows
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

```bash
# Linux / macOS
source venv/bin/activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

- Backend API: `http://127.0.0.1:8000`
- Interactive Swagger API Docs: `http://127.0.0.1:8000/docs`

#### Step 2: Start Frontend Application
Open a second terminal:

```bash
cd frontend
npm run dev
```

- Access the web dashboard at: `http://localhost:5173`

#### Features in the Web Dashboard:
- **Live Gaze & Pose Tracker:** Real-time driver webcam feed with face detection bounding box, 5 landmarks, eye center, and head pose orientation.
- **Attention Heatmap Viewer:** Overlay showing where the driver is looking in the forward scene.
- **Driver State Analytics:** Displays Attentive, Drowsy, or Distracted classifications with confidence metrics.
- **Single / Paired Image Upload:** Test custom driver face and road scene images directly from your computer.

---

### Method 2: Standalone Pipeline & Inference Scripts

You can test individual modules without running the web server:

#### Test Face Detector, Eye Geometry, and Gaze Estimation
```bash
python backend/detection/test_face_detector.py
```
*Outputs landmark metrics and creates a visualization test image.*

#### Test End-to-End Live Inference (Detector + Model)
```bash
python backend/detection/test_live_inference.py
```
*Loads driver and scene frames, extracts live 24-D gaze vector, runs model prediction, and verifies outputs.*

#### Single Sample Prediction
```bash
python predict.py
```
*Loads a sample from the dataset, runs inference using `checkpoints/best_model.pth`, prints predicted gaze coordinates, and generates `outputs/prediction_result.png`.*

#### Scan Real Driver State Over Temporal Windows
```bash
python scan_real_driver_state.py
python scan_temporal_windows.py
```

---

### Method 3: Model Training & Evaluation

#### 1. Train L-DriX-Net
Adjust hyperparameters in `config.py` (epochs, batch size, learning rate, dataset path), then run:

```bash
python train.py
```

- Automatically splits dataset (80% train, 20% validation).
- Computes Total Loss (MSE Loss, BCE Loss, and Cosine Similarity Loss).
- Saves the best checkpoint to `checkpoints/best_model.pth`.
- Generates loss progress plots saved to `training_curve.png`.

#### 2. Validate Trained Model
Evaluate metrics on the test/validation set:

```bash
python validate.py
```
- Computes AUC, Area-under-Curve, Mean Absolute Error (MAE), and MSE.
- Outputs evaluation metrics to `outputs/metrics.txt`.

---

### Method 4: Component Verification Tests

Run any of the included component tests to verify system subsystems:

```bash
# Verify configuration
python test_config.py

# Verify dataloader
python test_dataloader.py

# Verify model checkpoint loading
python test_checkpoint.py

# Verify L-DriX-Net forward pass
python test_ldrixnet.py

# Verify Temporal Attention module
python test_temporal_attention.py

# Verify Facial Encoder (LFEM)
python test_lfem.py

# Verify Scene Encoder
python test_scene_encoder.py

# Verify Cross-Fusion Module (ICFM)
python test_icfm.py

# Verify Heatmap Decoder
python test_decoder.py
```

---

## API Endpoints

The backend provides REST endpoints for live and batch processing:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API status and root welcome message. |
| `GET` | `/health` | Health check and status of loaded model checkpoint. |
| `POST` | `/predict` | Receives `driver_image` and `scene_image` (multipart form). Returns gaze coordinates, 24-D feature vector, driver state, and base64-encoded attention heatmap overlay. |

### Sample Prediction Request (cURL):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "driver_image=@path/to/driver_face.jpg" \
  -F "scene_image=@path/to/road_scene.jpg"
```

---

## Configuration

Core project parameters are centralized in `config.py`:

```python
# Dataset path
DATASET_PATH = "dataset/Subject01_1_data"

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Checkpoint path
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL = "checkpoints/best_model.pth"

# Device selection (Auto-selects CUDA if available)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Troubleshooting & FAQ

### 1. `YuNet model not found`
- Ensure `backend/models/face_detection_yunet_2026may.onnx` exists in your repository.
- Verify OpenCV is installed: `pip install opencv-python`.

### 2. `Checkpoint not found: checkpoints/best_model.pth`
- Place your trained checkpoint in the `checkpoints/` folder.
- If you do not have a checkpoint, train the model using `python train.py` to produce one.

### 3. Frontend Cannot Connect to Backend
- Confirm the backend is running on `http://127.0.0.1:8000`.
- Verify the `VITE_API_URL` environment variable or check `frontend/src/services/api.js`.
- Make sure CORS is enabled in `backend/main.py` for port 5173.

### 4. OpenCV or CUDA Memory Errors
- If out of GPU memory during training, reduce `BATCH_SIZE` in `config.py` from `8` to `4` or `2`.
- You can force CPU execution by setting `DEVICE = torch.device("cpu")` in `config.py`.

---

## License

This project is developed for research and educational purposes. See the repository license for additional terms.
