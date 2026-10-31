from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation, Input,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image
from transformers import pipeline

import numpy as np
import io
import zipfile
import os
import tempfile
import asyncio
import sqlite3

app = FastAPI(
    title="API Citrus Leaf Pathology",
    description="Clasificación de enfermedades de hojas de cítricos + recomendaciones LLM",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- LLM ----------
print("Cargando modelo LLM (Qwen)...")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
print("LLM cargado correctamente.")

def generar_recomendacion_llm(enfermedad_predicha: str) -> str:
    prompt = (
        f"Eres un agrónomo experto en fitopatología. Un sistema de inteligencia artificial "
        f"ha diagnosticado la siguiente condición en las hojas de la planta: '{enfermedad_predicha}'.\n"
        f"Por favor, redacta un informe detallado EN ESPAÑOL con recomendaciones de manejo, "
        f"tratamiento y prevención para esta enfermedad. La explicación debe ser clara, "
        f"profesional y tener una extensión mayor a 200 caracteres."
    )
    messages = [
        {"role": "system", "content": "Eres un asistente experto en agricultura y salud vegetal."},
        {"role": "user", "content": prompt},
    ]
    # do_sample=False → determinista (permite cachear y evita inventos)
    output = pipe(messages, max_new_tokens=300, do_sample=False)
    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        for msg in reversed(generated):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "").strip()
        last = generated[-1]
        return (last.get("content", "") if isinstance(last, dict) else str(last)).strip()

    return str(generated).strip()


# Candado: solo 1 generación simultánea (Qwen es pesado en CPU)
_llm_lock = asyncio.Semaphore(1)

async def generar_recomendacion_llm_async(enfermedad_predicha: str) -> str:
    async with _llm_lock:
        return await asyncio.to_thread(generar_recomendacion_llm, enfermedad_predicha)


# ---------- BD feedback RL ----------
def init_db():
    conn = sqlite3.connect("rl_feedback.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback_rl (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            enfermedad TEXT,
            recomendacion TEXT,
            rating INTEGER,
            reward REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

class FeedbackRequest(BaseModel):
    enfermedad: str
    recomendacion: str
    rating: int

@app.post("/api/feedback")
def guardar_feedback_rl(data: FeedbackRequest):
    if not (1 <= data.rating <= 5):
        raise HTTPException(status_code=400, detail="El rating debe ser entre 1 y 5")
    reward = (data.rating - 3) / 2.0
    conn = sqlite3.connect("rl_feedback.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO feedback_rl (enfermedad, recomendacion, rating, reward) VALUES (?, ?, ?, ?)",
        (data.enfermedad, data.recomendacion, data.rating, reward)
    )
    conn.commit()
    conn.close()
    return {"status": "success", "reward_calculado": reward}


# ---------- CNN ----------
MODEL_PATH = "modelo.keras"
IMG_SIZE = (224, 224)
L2 = 0.0002

CLASS_NAMES = [
    "Anthracnose", "Bacterial Blight", "Black Spot", "Citrus Canker",
    "Citrus Mite", "Curl Leaf", "Curl Virus", "Deficiency Leaf",
    "Dry leaf", "Greening", "Healthy", "Melanose", "Sooty Mould", "Spider Mite",
]  # 14

def build_model() -> Sequential:
    return Sequential([
        Input(shape=(224, 224, 3)),

        Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(), Activation("relu"), MaxPooling2D((2, 2)),

        Dropout(0.4),
        GlobalAveragePooling2D(),

        Dense(512, kernel_regularizer=l2(L2)),
        Activation("relu"),
        Dropout(0.5),

        Dense(14, activation="softmax"),
    ])

print("Construyendo arquitectura...")
model = build_model()

print("Extrayendo pesos del .keras...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(MODEL_PATH, "r") as z:
        weight_files = [
            f for f in z.namelist()
            if "weights" in f.lower() or f.endswith(".h5") or f.endswith(".weights.h5")
        ]
        if not weight_files:
            print("Contenido del .keras:", z.namelist())
            raise RuntimeError("No se encontraron pesos dentro del archivo .keras")
        weight_file = weight_files[0]
        print(f"Archivo de pesos encontrado: {weight_file}")
        z.extract(weight_file, tmpdir)
        model.load_weights(os.path.join(tmpdir, weight_file))

print("Modelo cargado correctamente.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img_array = img_to_array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {e}")


# ---------- Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """SOLO clasifica (rápido, ~1 s). No llama al LLM."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(predictions))
    top3_idx = predictions.argsort()[-3:][::-1]
    top3 = [{"clase": CLASS_NAMES[i], "confianza": float(predictions[i])} for i in top3_idx]

    return JSONResponse({
        "prediccion": CLASS_NAMES[idx],
        "confianza": float(predictions[idx]),
        "top_3": top3,
    })


class RecommendationRequest(BaseModel):
    enfermedad: str

_reco_cache: dict[str, str] = {}

@app.post("/recommendation")
async def recommendation(data: RecommendationRequest):
    """Genera recomendación con el LLM (lento, cacheada por enfermedad)."""
    if data.enfermedad not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Enfermedad no válida")

    if data.enfermedad in _reco_cache:
        return {
            "enfermedad": data.enfermedad,
            "recomendacion": _reco_cache[data.enfermedad],
            "cached": True,
        }

    try:
        reco = await generar_recomendacion_llm_async(data.enfermedad)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM falló: {e}")

    _reco_cache[data.enfermedad] = reco
    return {
        "enfermedad": data.enfermedad,
        "recomendacion": reco,
        "cached": False,
    }


@app.get("/health")
def health():
    return {"status": "ok"}