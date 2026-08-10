import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_WEIGHTS_PATH = os.path.join(ARTIFACTS_DIR, "genre_cnn.pt")
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")
MODEL_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "model_config.json")
PREPROCESSING_CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "preprocessing_config.json")
