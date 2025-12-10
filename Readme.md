# Face-Based Emotion Regression Tool

A project using MediaPipe FaceMesh, OpenCV, and a custom least-squares regression model to estimate an "emotion value" from facial landmark distances.

This repository contains three main scripts:

- `fit.py` – Runs the regression model and live webcam visualization
- `training.py` – Collects training samples (points + emotion values)
- `data_handler.py` – Utility for managing and cleaning your training data files

## Installation

This project requires Python 3.8+ and the following external libraries:

- OpenCV
- Mediapipe
- Numpy

### Install the Required Packages
```bash
pip install opencv-python mediapipe numpy
```

## training.py

Uses the webcam and MediaPipe Face Mesh to record feature vectors (distances between face landmarks) along with user-entered emotion values. These samples are saved to a `.pkl` dataset file.

### How to Run:

```bash
python training.py
```

### Inputs Required:

From keyboard while script runs:

 #### Hold `c` → capture a sample

- Script asks: “Enter emotion value between 0 (Neutral) and 1 (Happy):”

 #### Hold `q` → stop capturing

- Script asks: “Enter filename to save data:”

### Output:

A `.pkl` file containing:
- `points`: list of feature vectors
- `values`: list of corresponding emotion labels

## fit.py
Loads a dataset file created by `training.py`, computes a linear regression model (least squares), then runs real-time emotion prediction and visualization using the webcam.

How to Run:

```bash
python fit.py
```

### Inputs Required:

At program start:

- Filename of an existing `.pkl` dataset (produced by `training.py`) where number of samples ≥2.

### data_handler.py

A utility for inspecting, combining, and cleaning `.pkl` data files produced by `training.py`.

How to Run:

```bash
python data_handler.py
```