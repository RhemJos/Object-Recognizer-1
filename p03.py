from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# results = model("monos.jpg")
results = model.predict(source="perros.jpg")

# Obtener todas las predicciones
detections = results[0]  # La primera predicción (si solo usas una imagen)

dog_class_id = [key for key, value in model.names.items() if value == 'dog'][0]

dogs = [box for box in detections.boxes if int(box.cls) == dog_class_id]

# Mostrar los resultados filtrados
print(f"Se encontraron {len(dogs)} perros en la imagen.")


