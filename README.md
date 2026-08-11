# 🎵 GenreGuess: AI Music Genre Classification

An end-to-end Machine Learning and Deep Learning system for classifying 30-second audio clips into one of 10 musical genres using the GTZAN dataset. 

This project explores and compares two distinct approaches to audio classification:
1. **Classical Machine Learning**: Utilizes pre-extracted tabular features (like MFCCs, chroma, and spectral centroids) fed into Logistic Regression and Random Forest models.
2. **Deep Learning**: Learns representations directly from Log-Mel Spectrograms (2D time-frequency images) using a custom Convolutional Neural Network (CNN) trained from scratch in PyTorch.

## 🚀 Live Demo & API
The model is served via a **FastAPI** backend and is fully containerized for deployment.

### Endpoints
**`GET /health`**
Returns the status of the API and verifies if the PyTorch model loaded successfully.
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**`POST /predict`**
Accepts a `.wav` file upload and returns the predicted genre and confidence.
```bash
curl -X POST -F "file=@test.wav" http://localhost:8000/predict
```

## 🧠 Design Decisions & Architecture

1. **Why a CNN?** Spectrograms exhibit local spatial structure (e.g., strong low-frequency energy for hiphop, harmonic lines for classical). CNNs exploit this local connectivity via weight sharing, making them highly parameter-efficient for audio representations compared to fully-connected networks.
2. **Why Mel Spectrograms?** The mel scale approximates human pitch perception, compressing the frequency axis meaningfully. It reduces dimensionality while preserving timbre and rhythm, making it far superior to raw waveforms which require massive datasets to learn Fourier-like transformations.
3. **Robust Data Splitting**: We use a stratified split at the track level to prevent segments of the same song from leaking across training and validation sets. A hash-based duplicate check is included to identify known GTZAN dataset duplicates.
4. **CNN Architecture**: A 4-block CNN with Batch Normalization and ReLU, topped with an `AdaptiveAvgPool2d`. The adaptive pooling ensures the network is robust to small variations in audio length without hardcoding flatten dimensions. We apply dropout in the fully connected head to control overfitting.

## 🛠️ Repository Structure

```
genreguess/
├── data/                  # Raw and processed datasets (spectrograms)
├── notebooks/             # EDA and classical baselines
├── src/                   # Source code for ML pipelines and PyTorch models
├── scripts/               # Helper scripts for data download and preprocessing
├── artifacts/             # Exported PyTorch weights & JSON configs
├── reports/               # Training metrics and confusion matrices
├── app/                   # FastAPI inference service
├── tests/                 # Pytest test suite
├── Dockerfile             # Containerization config
└── requirements.txt       # Python dependencies
```

## 💻 Setup & Reproduction

1. **Environment Setup**:
   Create a virtual environment using Python 3.11:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Dataset Preparation**:
   Place the GTZAN `.wav` files into `data/raw/genres_original/`. If you don't have them, run `python scripts/download_data.py`. Ensure `features_30_sec.csv` is in `data/raw/`.
3. **Precompute Spectrograms**:
   ```bash
   python scripts/precompute_spectrograms.py
   ```
4. **Train Models**:
   - Classical Models: `python src/classical_ml/train_baselines.py`
   - PyTorch CNN: `python scripts/run_experiments.py`
5. **Run the API Locally**:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

## ☁️ Deployment
The service is fully containerized using Docker. PyTorch is pinned to CPU-only in the requirements to keep the image lightweight (inference on a single 30s spectrogram takes milliseconds on a CPU). It is designed to be easily deployed on platforms like Render or AWS ECS.