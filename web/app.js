const API_URL = `${ENV.BACKEND_URL}/predict`;
const RECO_URL = `${ENV.BACKEND_URL}/recommendation`;
const FEEDBACK_URL = `${ENV.BACKEND_URL}/api/feedback`;

const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const analyzeBtn = document.getElementById('analyzeBtn');
const llmPanel = document.getElementById('llmPanel');
const resultBox = document.getElementById('resultBox');
const resultTitle = document.getElementById('resultTitle');
const resultDetail = document.getElementById('resultDetail');
const llmRecommendationText = document.getElementById('llmRecommendationText');
const stars = document.querySelectorAll('.star');
const feedbackMsg = document.getElementById('feedbackMsg');

let selectedFile = null;
let ultimaEnfermedad = "";
let ultimaRecomendacion = "";
let ratingSeleccionado = 0;

// 1. Selección de imagen
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (event) => {
            previewImage.src = event.target.result;
            previewContainer.style.display = 'block';
            analyzeBtn.style.display = 'block';
            resultBox.style.display = 'none';
            llmPanel.style.display = 'none';
            resultBox.classList.remove('error');
        };
        reader.readAsDataURL(file);
    }
});

// 2. Analizar: dos llamadas (CNN rápida → LLM lenta)
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) {
        alert("Selecciona una imagen primero");
        return;
    }

    // Reset UI
    llmPanel.style.display = 'flex';
    resultBox.style.display = 'block';
    resultBox.classList.remove('error');
    resultTitle.textContent = "Procesando...";
    resultDetail.textContent = "Analizando imagen con CNN...";
    llmRecommendationText.textContent = "Esperando diagnóstico para generar recomendación...";
    analyzeBtn.disabled = true;
    ultimaEnfermedad = "";
    ultimaRecomendacion = "";

    // ---- PASO 1: Predicción (rápida) ----
    try {
        const formData = new FormData();
        formData.append("file", selectedFile);

        const predResp = await fetch(API_URL, { method: "POST", body: formData });
        const predData = await predResp.json();
        if (!predResp.ok) throw new Error(predData.detail || "Error del servidor");

        const enfermedad = predData.prediccion || "No identificada";
        const confianza = predData.confianza
            ? (predData.confianza * 100).toFixed(1) + "%"
            : "N/A";

        ultimaEnfermedad = enfermedad;

        resultTitle.textContent = "Diagnóstico";
        resultDetail.innerHTML = `
            <strong>Enfermedad detectada:</strong> ${enfermedad}<br>
            <strong>Nivel de confianza:</strong> ${confianza}
        `;
    } catch (error) {
        console.error(error);
        resultBox.classList.add('error');
        resultTitle.textContent = "Error de conexión";
        resultDetail.textContent = "No se pudo conectar con la API. ¿Está corriendo el backend?";
        llmRecommendationText.textContent = "No disponible.";
        analyzeBtn.disabled = false;
        return; // sin predicción no tiene sentido pedir recomendación
    }

    // ---- PASO 2: Recomendación LLM (lenta, con caché en backend) ----
    llmRecommendationText.textContent =
        `Generando recomendación agronómica para "${ultimaEnfermedad}"... (puede tardar unos segundos)`;

    try {
        const recoResp = await fetch(RECO_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ enfermedad: ultimaEnfermedad })
        });
        const recoData = await recoResp.json();
        if (!recoResp.ok) throw new Error(recoData.detail || "Error generando recomendación");

        ultimaRecomendacion = recoData.recomendacion || "Sin recomendación generada.";
        llmRecommendationText.textContent = ultimaRecomendacion;

        // Reiniciar estrellas de feedback
        stars.forEach(s => s.classList.remove('selected'));
        feedbackMsg.style.display = 'none';
        ratingSeleccionado = 0;
    } catch (error) {
        console.error(error);
        llmRecommendationText.textContent =
            "No se pudo generar la recomendación. Intenta de nuevo.";
    } finally {
        analyzeBtn.disabled = false;
    }
});

// 3. Feedback RL (estrellas) — sin cambios funcionales
stars.forEach(star => {
    star.addEventListener('click', async () => {
        const rating = parseInt(star.getAttribute('data-value'));
        ratingSeleccionado = rating;

        stars.forEach(s => {
            if (parseInt(s.getAttribute('data-value')) <= rating) {
                s.classList.add('selected');
            } else {
                s.classList.remove('selected');
            }
        });

        if (!ultimaEnfermedad || !ultimaRecomendacion) {
            alert("Primero espera a que se genere la recomendación antes de enviar feedback.");
            return;
        }

        try {
            const res = await fetch(FEEDBACK_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    enfermedad: ultimaEnfermedad,
                    recomendacion: ultimaRecomendacion,
                    rating: ratingSeleccionado
                })
            });
            const resultData = await res.json();
            if (res.ok) {
                feedbackMsg.style.display = 'block';
                feedbackMsg.textContent =
                    `¡Feedback registrado! Recompensa RL: ${resultData.reward_calculado}`;
            } else {
                alert("Error al guardar el feedback.");
            }
        } catch (e) {
            console.error("Error de red al enviar feedback:", e);
        }
    });
});