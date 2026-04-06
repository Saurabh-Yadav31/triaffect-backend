import os
import json
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

audio_model = None
facial_model = None
text_model = None
text_tokenizer = None
audio_labels = None
facial_labels = None
text_labels = None

def strip_quantization_config(config):
    """Recursively remove quantization_config from layer configs."""
    if isinstance(config, dict):
        config.pop("quantization_config", None)
        for key in config:
            config[key] = strip_quantization_config(config[key])
    elif isinstance(config, list):
        config = [strip_quantization_config(item) for item in config]
    return config

def load_model_from_json_and_weights(arch_path, weights_path):
    with open(arch_path) as f:
        arch = json.load(f)
    arch = strip_quantization_config(arch)
    model = tf.keras.models.model_from_json(json.dumps(arch))
    model.load_weights(weights_path)
    return model

def load_all_models():
    global audio_model, facial_model, text_model, text_tokenizer
    global audio_labels, facial_labels, text_labels

    # ── AUDIO ──
    print("Loading audio model...")
    audio_model = load_model_from_json_and_weights(
        os.path.join(MODELS_DIR, "audio", "audio_arch.json"),
        os.path.join(MODELS_DIR, "audio", "audio_weights.weights.h5")
    )
    with open(os.path.join(MODELS_DIR, "audio", "audio_label_config.json")) as f:
        audio_labels = json.load(f)
    print("✅ Audio model loaded!")

    # ── FACIAL ──
    print("Loading facial model...")
    facial_model = load_model_from_json_and_weights(
        os.path.join(MODELS_DIR, "facial", "facial_arch.json"),
        os.path.join(MODELS_DIR, "facial", "facial_weights.weights.h5")
    )
    with open(os.path.join(MODELS_DIR, "facial", "label_config.json")) as f:
        facial_labels = json.load(f)
    print("✅ Facial model loaded!")

    # ── TEXT ──
    print("Loading text model...")
    text_tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODELS_DIR, "text"))
    text_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODELS_DIR, "text"))
    text_model.eval()
    with open(os.path.join(MODELS_DIR, "text", "label_config.json")) as f:
        text_labels = json.load(f)
    print("✅ Text model loaded!")

    print("\n🎉 All models loaded successfully!")