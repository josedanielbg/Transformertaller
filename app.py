# app.py

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import gradio as gr
import os

# It's good practice to get the token from environment variables in deployment
# For local testing, you might set it directly, but remove for public spaces.
# hf_token = os.environ.get('HF_TOKEN')

# Cargar el procesador y el modelo
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Función para procesar la imagen
def detect_objects(image):
    # Preprocesamiento
    inputs = processor(images=image, return_tensors="pt")

    # Detectar objetos
    with torch.no_grad():
        outputs = model(**inputs)

    # Filtrar resultados
    target_sizes = torch.tensor([image.size[::-1]])  # (alto, ancho)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # Crear una lista de los resultados con nombre y puntuación
    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detected_objects.append(f"Objeto: {model.config.id2label[label.item()]}, Score: {score:.2f}, Box: {box.tolist()}")

    if not detected_objects:
        return "No objects detected with threshold 0.9."
    return "\n".join(detected_objects)

# Crear la interfaz Gradio
interface = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(),
    live=True,
    title="Detección de Objetos con Transformers",
    description="Sube una imagen y descubre qué objetos se pueden detectar."
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()
