# Music Genre Classification

An end-to-end Machine Learning and Deep Learning system for classifying 30-second audio clips into one of 10 musical genres using the GTZAN dataset. 

This project explores and compares two distinct approaches to audio classification:
1. **Classical ML (Track A)**: Utilizes pre-extracted tabular features (like MFCCs, chroma, and spectral centroids) fed into Logistic Regression and Random Forest models.
2. **Deep Learning (Track B)**: Learns representations directly from Log-Mel Spectrograms (2D time-frequency images) using a custom Convolutional Neural Network (CNN) trained from scratch in PyTorch.

## Problem Statement
The goal is to accurately classify audio tracks into 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock). This poses a genuine multi-class classification challenge that requires careful handling of signal processing, preventing data leakage, and training stability.

## Repository Structure

```
genreguess/
├── data/
│   ├── raw/                       # GTZAN audio + features_30_sec.csv
│   ├── processed/spectrograms/    # Precomputed .npy mel spectrograms
│   └── splits/                    # Train/Val/Test JSON splits
├── notebooks/
│   └── classical_ml_eda.ipynb     # Original EDA and classical baselines
├── src/
│   ├── classical_ml/              # Pipeline for RF & LR baselines
│   ├── deep_learning/             # PyTorch CNN, dataset, and training loop
│   └── common/                    # Shared split utils and audio preprocessing
├── scripts/
│   ├── download_data.py           # HF Datasets download script
│   ├── precompute_spectrograms.py # Batch conversion script
│   └── run_experiments.py         # Hyperparameter and augmentation experiments
├── artifacts/                     # Exported PyTorch weights & JSON configs
├── reports/                       # Confusion matrices & metrics
├── app/                           # FastAPI inference service
├── tests/                         # Pytest suite
├── Dockerfile
└── requirements.txt
```

## Setup & Reproduction

1. **Environment Setup**:
   Create a virtual environment using Python 3.11 or 3.12 (Python 3.14 is currently not supported for `scikit-learn` compilation without C++ Build Tools).
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. **Dataset**:
   Place the GTZAN `.wav` files into `data/raw/genres_original/`. If you don't have them, you can try running `python scripts/download_data.py`. Ensure `features_30_sec.csv` is in `data/raw/`.
3. **Precompute Spectrograms**:
   ```bash
   python scripts/precompute_spectrograms.py
   ```
4. **Train Models**:
   - Classical: `python src/classical_ml/train_baselines.py`
   - CNN: `python scripts/run_experiments.py`

## Results & Execution Note

> [!WARNING]
> **Execution Status**: The implementation is fully written, but during execution on the development machine (Windows, Python 3.14), `scikit-learn` failed to compile due to missing Microsoft C++ Build Tools 14.0+. Additionally, the `datasets` module could not be installed. As per project constraints, **no metrics have been fabricated**. 
> To obtain the final accuracy, F1-scores, and confusion matrices, please run the reproduction steps above in an environment with Python 3.11/3.12 or with C++ Build Tools installed. The reports and artifacts directories will populate upon successful runs.

## Design Decisions

1. **Why CNN?** Spectrograms exhibit local spatial structure (e.g., strong low-frequency energy for hiphop, harmonic lines for classical). CNNs exploit this local connectivity via weight sharing, making them highly parameter-efficient for audio representations compared to fully-connected networks.
2. **Why Mel Spectrogram?** The mel scale approximates human pitch perception, compressing the frequency axis meaningfully. It reduces dimensionality while preserving timbre and rhythm, making it far superior to raw waveforms which require massive datasets to learn Fourier-like transformations.
3. **Train/Val/Test Split**: We use a stratified split at the track level to prevent segments of the same song from leaking across splits. A hash-based duplicate check is included in `split_utils.py` to identify known GTZAN duplicates.
4. **CNN Architecture**: A 4-block CNN with Batch Normalization and ReLU, topped with an `AdaptiveAvgPool2d`. The adaptive pooling ensures the network is robust to small variations in audio length without hardcoding flatten dimensions. We apply dropout only in the fully connected head (p=0.4) to control overfitting on this small (1000 sample) dataset.
5. **No Pretrained Models**: We train from scratch because there is no ubiquitous ImageNet-equivalent vision backbone perfectly tuned for log-mel spectrograms. Training from scratch also explicitly demonstrates end-to-end understanding of backpropagation, architecture design, and regularization.

## API Documentation

The FastAPI service exposes a stateless inference backend.

### Run Locally
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

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
```json
{
  "genre": "blues",
  "confidence": 0.952,
  "probabilities": {
    "blues": 0.952,
    "rock": 0.048,
    "...": "..."
  }
}
```

## Deployment

The service is fully containerized and designed for deployment to a managed platform like **Render** or **Railway**.

1. The `Dockerfile` uses `python:3.11-slim` and installs `libsndfile1` (required by librosa/soundfile).
2. PyTorch is pinned to CPU-only in `requirements.txt` (`--extra-index-url https://download.pytorch.org/whl/cpu`) to keep the Docker image lightweight. Inference on a single 30s spectrogram takes milliseconds on a CPU, making GPU provisioning unnecessary.
3. The model weights (`.pt`) and configs are copied directly into the image.
4. **Deployment Steps (Render)**:
   - Connect the GitHub repository to Render.
   - Choose "Docker" as the environment.
   - Render will build the image and expose port `8000`. No database or state management is required.

## Known Limitations
- GTZAN is known to have exact and near-duplicate tracks, which can inflate test accuracy if not perfectly handled. Our `split_utils.py` logs exact SHA256 matches, but audio-fingerprinting is required to catch all near-duplicates.
- The CNN is trained on a very small dataset (~700 training samples) and is susceptible to overfitting. SpecAugment (frequency and time masking) is implemented to combat this.