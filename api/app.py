from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import json

# ------------------------------------------------------------------
# Cargar modelo y mapeo de clases
# ------------------------------------------------------------------
#MODEL_PATH = "mejor_modelo_hojas.keras"
MODEL_PATH = "modelo.keras"
CLASS_MAP_PATH = "class_indices.json"

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_MAP_PATH) as f:
    class_map = {int(k): v for k, v in json.load(f).items()}

# ------------------------------------------------------------------
# App FastAPI
# ------------------------------------------------------------------
app = FastAPI(
    title="API Diagnóstico de Hojas de Limón",
    description="Clasificación de enfermedades en hojas de limón",
    version="1.0"
)

# CORS (útil si vas a consumir la API desde frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "mensaje": "API de diagnóstico de hojas de limón activa",
        "clases": list(class_map.values()),
        "endpoint_prediccion": "/predict"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validar que sea una imagen
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")

    try:
        # Leer imagen
        contents = await file.read()
        npimg = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Imagen inválida")

        # Preprocesamiento (igual que en tu código Flask)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_resized, axis=0)

        # Predicción
        pred = model.predict(img_batch, verbose=0)[0]
        idx = int(np.argmax(pred))
        confianza = round(float(pred[idx] * 100), 2)
        clase = class_map[idx]

        # Top 3 (opcional, pero útil)
        top3_idx = pred.argsort()[-3:][::-1]
        top3 = [
            {
                "clase": class_map[i],
                "confianza_porcentaje": round(float(pred[i] * 100), 2)
            }
            for i in top3_idx
        ]           

        return JSONResponse({
            "prediccion": clase,
            "confianza": round(float(pred[idx]), 4), 
            "indice": idx,
            "top_3": top3
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la imagen: {str(e)}")