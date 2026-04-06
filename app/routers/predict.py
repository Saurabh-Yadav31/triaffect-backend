from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
import soundfile as sf
import librosa
import tempfile
import os
from io import BytesIO
from PIL import Image

router = APIRouter()

# ── Load YAMNet once at module level ──────────────────────────────────────────
print("Loading YAMNet for audio feature extraction...")
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
print("YAMNet loaded!")

AUDIO_LABELS = ["angry", "happy", "neutral", "sad", "stressed"]

# ── TEXT ──────────────────────────────────────────────────────────────────────

class TextRequest(BaseModel):
    text: str


@router.post("/text")
def predict_text(request: TextRequest):
    from app.utils.model_loader import text_model, text_tokenizer, text_labels
    if text_model is None:
        raise HTTPException(status_code=503, detail="Text model not loaded")

    inputs = text_tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    )
    with torch.no_grad():
        outputs = text_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()

    if isinstance(probs, float):
        probs = [probs]

    labels = text_labels.get("id2label", {str(i): str(i) for i in range(len(probs))})
    result = {labels[str(i)]: round(probs[i], 4) for i in range(len(probs))}
    top_emotion = max(result, key=result.get)

    return {
        "modality": "text",
        "input": request.text,
        "top_emotion": top_emotion,
        "confidence": result[top_emotion],
        "all_emotions": result
    }


# ── FACE ──────────────────────────────────────────────────────────────────────

@router.post("/face")
def predict_face(file: UploadFile = File(...)):
    from app.utils.model_loader import facial_model, facial_labels
    if facial_model is None:
        raise HTTPException(status_code=503, detail="Facial model not loaded")

    try:
        contents = file.file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img = img.resize((96, 96))                              # ✅ Fixed: was 224
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

        preds = facial_model.predict(img_array, verbose=0)[0]
        probs = preds.tolist()

        labels = facial_labels.get("id2label", {str(i): str(i) for i in range(len(probs))})
        result = {labels[str(i)]: round(probs[i], 4) for i in range(len(probs))}
        top_emotion = max(result, key=result.get)

        return {
            "modality": "face",
            "top_emotion": top_emotion,
            "confidence": result[top_emotion],
            "all_emotions": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── AUDIO ─────────────────────────────────────────────────────────────────────

@router.post("/audio")
async def predict_audio(file: UploadFile = File(...)):
    from app.utils.model_loader import audio_model, audio_labels
    if audio_model is None:
        raise HTTPException(status_code=503, detail="Audio model not loaded")

    contents = await file.read()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Read audio
        audio_np, sample_rate = sf.read(tmp_path)

        # Convert stereo → mono
        if audio_np.ndim > 1:
            audio_np = np.mean(audio_np, axis=1)

        # Resample to 16kHz (YAMNet requirement)
        if sample_rate != 16000:
            audio_np = librosa.resample(audio_np, orig_sr=sample_rate, target_sr=16000)

        audio_np = audio_np.astype(np.float32)

        # Normalize (same as training)
        audio_np = audio_np / (np.max(np.abs(audio_np)) + 1e-9)

        # ✅ Extract YAMNet embeddings (mean + std = 2048-dim, exactly as trained)
        waveform = tf.constant(audio_np, dtype=tf.float32)
        _, embeddings, _ = yamnet_model(waveform)
        emb = embeddings.numpy()

        mean_emb = np.mean(emb, axis=0)                        # (1024,)
        std_emb  = np.std(emb,  axis=0)                        # (1024,)
        feature_vec = np.concatenate([mean_emb, std_emb]).reshape(1, -1)  # (1, 2048)

        # Predict
        preds = audio_model.predict(feature_vec, verbose=0)[0]
        probs = preds.tolist()

        # Use audio_labels if available, else fallback to hardcoded order from training
        if audio_labels and "id2label" in audio_labels:
            labels = audio_labels["id2label"]
            result = {labels[str(i)]: round(probs[i], 4) for i in range(len(probs))}
        else:
            result = {AUDIO_LABELS[i]: round(probs[i], 4) for i in range(len(probs))}

        top_emotion = max(result, key=result.get)

        return {
            "modality": "audio",
            "top_emotion": top_emotion,
            "confidence": result[top_emotion],
            "all_emotions": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        os.unlink(tmp_path)


# ── COMBINED ──────────────────────────────────────────────────────────────────

@router.post("/combined")
async def predict_combined(
    text: Optional[str] = None,
    face_file: Optional[UploadFile] = File(None),
    audio_file: Optional[UploadFile] = File(None)
):
    results = {}
    weights = {}

    if text:
        text_result = predict_text(TextRequest(text=text))
        results["text"] = text_result["all_emotions"]
        weights["text"] = 0.4

    if face_file:
        face_result = predict_face(face_file)
        results["face"] = face_result["all_emotions"]
        weights["face"] = 0.35

    if audio_file:
        audio_result = await predict_audio(audio_file)
        results["audio"] = audio_result["all_emotions"]
        weights["audio"] = 0.25

    if not results:
        raise HTTPException(status_code=400, detail="At least one input required")

    # Collect all emotion labels
    all_emotions = set()
    for r in results.values():
        all_emotions.update(r.keys())

    # Adaptive weighted fusion
    total_weight = sum(weights.values())
    fused = {}
    for emotion in all_emotions:
        score = 0.0
        for modality, probs in results.items():
            score += weights[modality] * probs.get(emotion, 0.0)
        fused[emotion] = round(score / total_weight, 4)

    top_emotion = max(fused, key=fused.get)

    return {
        "modality": "combined",
        "inputs_used": list(results.keys()),
        "top_emotion": top_emotion,
        "confidence": fused[top_emotion],
        "all_emotions": fused,
        "individual_results": results
    }