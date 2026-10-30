
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Activation, Input
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import zipfile
import os
import tempfile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import sqlite3
import tensorflow as tf  # O la librería que uses para tu CNN

app = FastAPI()

# --- 1. CARGAR EL LLM CON PIPELINE (Qwen2.5-1.5B-Instruct) ---
print("Cargando modelo LLM (Qwen)...")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
print("LLM cargado correctamente.")

def generar_recomendacion_llm(enfermedad_predicha: str) -> str:
    prompt = f"""
    Eres un agrónomo experto en fitopatología. Un sistema de inteligencia artificial ha diagnosticado la siguiente condición en las hojas de la planta: '{enfermedad_predicha}'.
    Por favor, redacta un informe detallado con recomendaciones de manejo, tratamiento y prevención para esta enfermedad. La explicación debe ser clara, profesional y tener una extensión mayor a 200 caracteres.
    """
    
    messages = [
        {"role": "system", "content": "Eres un asistente experto en agricultura y salud vegetal."},
        {"role": "user", "content": prompt}
    ]
    
    # Generar texto con la tubería
    output = pipe(messages, max_new_tokens=300, do_sample=True, temperature=0.7)
    return output[0]['generated_text'][-1]['content']

# --- 2. CONFIGURACIÓN DE BASE DE DATOS PARA RL (FEEDBACK) ---
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
    rating: int  # Escala de 1 a 5

@app.post("/api/feedback")
def guardar_feedback_rl(data: FeedbackRequest):
    if not (1 <= data.rating <= 5):
        raise HTTPException(status_code=400, detail="El rating debe ser entre 1 y 5")
    
    # Recompensa para Aprendizaje por Refuerzo (RL) mapeada a [-1.0, 1.0]
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

# --- 3. ENDPOINT DE PREDICCIÓN CONECTADO AL LLM ---
@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    # Aquí va la lógica de tu CNN procesando la imagen y obteniendo la clase/enfermedad
    enfermedad_predicha = "Mancha bacteriana" # Ejemplo simulado de tu modelo CNN
    
    # Tubería: La predicción entra como INPUT al LLM para generar el texto
    recomendacion_texto = generar_recomendacion_llm(enfermedad_predicha)
    
    return {
        "enfermedad": enfermedad_predicha,
        "recomendacion": recomendacion_texto,
        "longitud_caracteres": len(recomendacion_texto)
    }


MODEL_PATH = "modelo.keras"          
IMG_SIZE = (224, 224)
L2 = 0.0002

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Blight",
    "Black Spot",
    "Citrus Canker",
    "Citrus Mite",
    "Curl Leaf",
    "Curl Virus",
    "Deficiency Leaf",
    "Dry leaf",
    "Greening",
    "Healthy",
    "Melanose",
    "Sooty Mould",
    "Spider Mite"
]

# Arquitectura de la red
def build_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),

        Conv2D(64, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),

        Conv2D(256, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),

        Conv2D(512, (3, 3), padding="same", kernel_regularizer=l2(L2)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D((2, 2)),

        Dropout(0.4),
        GlobalAveragePooling2D(),

        Dense(512, kernel_regularizer=l2(L2)),
        Activation("relu"),
        Dropout(0.5),

        Dense(13, activation="softmax")
    ])
    return model

# ------------------------------------------------------------------
# Cargar pesos desde el archivo .keras (es un zip)
# ------------------------------------------------------------------
print("Construyendo arquitectura...")
model = build_model()

print("Extrayendo pesos del .keras...")
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(MODEL_PATH, "r") as z:
        # Buscar el archivo de pesos dentro del .keras
        weight_files = [f for f in z.namelist() if "weights" in f.lower() or f.endswith(".h5") or f.endswith(".weights.h5")]
        if not weight_files:
            # Listar contenido para depuración
            print("Contenido del .keras:", z.namelist())
            raise RuntimeError("No se encontraron pesos dentro del archivo .keras")
        
        weight_file = weight_files[0]
        print(f"Archivo de pesos encontrado: {weight_file}")
        z.extract(weight_file, tmpdir)
        weights_path = os.path.join(tmpdir, weight_file)
        
        model.load_weights(weights_path)

print("Modelo cargado correctamente.")


app = FastAPI(
    title="API Citrus Leaf Pathology",
    description="Clasificación de enfermedades de hojas de cítricos",
    version="1.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la imagen: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
def root():
    return {
        "mensaje": "API de clasificación de hojas de cítricos activa",
        "clases": CLASS_NAMES,
        "endpoint_prediccion": "/predict"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx])

    top3_idx = predictions.argsort()[-3:][::-1]
    top3 = [
        {"clase": CLASS_NAMES[i], "confianza": float(predictions[i])}
        for i in top3_idx
    ]

    return JSONResponse({
        "prediccion": CLASS_NAMES[predicted_idx],
        "confianza": confidence,
        "top_3": top3
    })

@app.get("/health")
def health():
    return {"status": "ok"}

