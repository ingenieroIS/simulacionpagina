 const API_URL = `${ENV.BACKEND_URL}/predict`;


 const fileInput = document.getElementById('fileInput');
 const previewContainer = document.getElementById('previewContainer');
 const previewImage = document.getElementById('previewImage');
 const analyzeBtn = document.getElementById('analyzeBtn');
 const resultBox = document.getElementById('resultBox');
 const resultTitle = document.getElementById('resultTitle');
 const resultDetail = document.getElementById('resultDetail');

 let selectedFile = null;

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

     resultBox.style.display = 'block';
     resultBox.classList.remove('error');
     resultTitle.textContent = "Procesando...";
     resultDetail.textContent = "Enviando imagen a la red neuronal...";
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
             const enfermedad = data.prediccion || "No identificada";
             const confianza = data.confianza
                 ? (data.confianza * 100).toFixed(1) + "%"
                 : "N/A";

             resultDetail.innerHTML = `
   <strong>Enfermedad detectada:</strong> ${enfermedad}<br>
   <strong>Nivel de confianza:</strong> ${confianza}
 `;
         } else {
             throw new Error(data.detail || "Error del servidor");
         }
     } catch (error) {
         console.error(error);
         resultBox.classList.add('error');
         resultTitle.textContent = "Error de conexión";
         resultDetail.textContent = "No se pudo conectar con la API. ¿Está corriendo en el puerto 8000?";
     } finally {
         analyzeBtn.disabled = false;
     }
 });