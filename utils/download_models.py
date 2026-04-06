from huggingface_hub import snapshot_download
import os

MODELS = {
    "audio": "Saurabh-Yadav31/triaffect-audio-model",
    "facial": "Saurabh-Yadav31/triaffect-facial-model",
    "text": "Saurabh-Yadav31/triaffect-text-model"
}

os.makedirs("models", exist_ok=True)

for name, repo in MODELS.items():
    print(f"Downloading {name}...")
    snapshot_download(repo_id=repo, local_dir=f"models/{name}")