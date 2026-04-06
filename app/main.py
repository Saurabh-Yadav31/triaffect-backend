from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.utils.model_loader import load_all_models
from app.routers import predict

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting TriAffect backend...")
    load_all_models()
    yield
    print("🛑 Shutting down TriAffect backend...")

app = FastAPI(
    title="TriAffect API",
    description="Tri-modal Emotion Recognition: Text + Audio + Facial",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://triaffect.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "TriAffect API is running"}