const API_URL = `${ENV.BACKEND_URL}/predict`;
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

 // 2. Envío a la API
 analyzeBtn.addEventListener('click', async () => {
     if (!selectedFile) {
         alert("Selecciona una imagen primero");
         return;
     }

     llmPanel.style.display = 'flex';
     resultBox.style.display = 'block';
     resultBox.classList.remove('error');
     resultTitle.textContent = "Procesando...";
     resultDetail.textContent = "Analizando imagen con CNN y generando reporte con LLM...";
     llmRecommendationText.textContent = "Generando texto agronómico...";
     analyzeBtn.disabled = true;

     try {
         const formData = new FormData();
         formData.append("file", selectedFile);  // ← campo que espera tu API

         const response = await fetch(API_URL, {
             method: "POST",
             body: formData
         });

         const data = await response.json();

         if (response.ok) {
             resultTitle.textContent = "Diagnóstico Final";

             // Claves que devuelve TU API (en español)
             const enfermedad = data.enfermedad || data.prediccion || "No identificada";
             const confianza = data.confianza
                 ? (data.confianza * 100).toFixed(1) + "%"
                 : "N/A";

                 ultimaEnfermedad = enfermedad;
                 ultimaRecomendacion = data.recomendacion || "Sin recomendación generada.";
                 resultDetail.innerHTML = `
                    <strong>Enfermedad detectada:</strong> ${enfermedad}<br>
                    <strong>Nivel de confianza:</strong> ${confianza}
                  `;
             // Mostrar texto generativo del LLM (>200 caracteres)
             llmRecommendationText.textContent = ultimaRecomendacion;

             // Reiniciar estrellas de feedback
             stars.forEach(s => s.classList.remove('selected'));
             feedbackMsg.style.display = 'none';
             ratingSeleccionado = 0;
         } else {
             throw new Error(data.detail || "Error del servidor");
         }
     } catch (error) {
         console.error(error);
         resultBox.classList.add('error');
         resultTitle.textContent = "Error de conexión";
         resultDetail.textContent = "No se pudo conectar con la API. ¿Está corriendo en el puerto 8000?";
         llmRecommendationText.textContent = "No disponible.";
     } finally {
         analyzeBtn.disabled = false;
     }
 });
 // 3. Sistema de Feedback por Aprendizaje por Refuerzo (Estrellas)
 stars.forEach(star => {
     star.addEventListener('click', async () => {
         const rating = parseInt(star.getAttribute('data-value'));
         ratingSeleccionado = rating;

         // Pintar estrellas seleccionadas
         stars.forEach(s => {
             if (parseInt(s.getAttribute('data-value')) <= rating) {
                 s.classList.add('selected');
             } else {
                 s.classList.remove('selected');
             }
         });

         if (!ultimaEnfermedad || !ultimaRecomendacion) {
             alert("Primero debes realizar un diagnóstico antes de enviar feedback.");
             return;
         }

         // Enviar feedback al endpoint de RL en FastAPI
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
                 feedbackMsg.textContent = `¡Feedback registrado! Recompensa RL: ${resultData.reward_calculado}`;
             } else {
                 alert("Error al guardar el feedback.");
             }
         } catch (e) {
             console.error("Error de red al enviar feedback:", e);
         }
     });
 });

