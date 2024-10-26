from ultralytics import YOLO

# Cargar modelo YOLO preentrenado
model = YOLO("yolo11n.pt")

results = model.predict(source="perros.jpg")

# Obtener todas las predicciones
detections = results[0]  # La primera predicción (si solo usas una imagen)

person_class_id = [key for key, value in model.names.items() if value == 'person'][0]

persons = [box for box in detections.boxes if int(box.cls) == person_class_id]

# Mostrar los resultados filtrados
print(f"Se encontraron {len(persons)} personas en la imagen.")


