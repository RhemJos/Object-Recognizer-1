from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado
model = YOLO("yolo11n.pt")

# Realizar la predicción en una imagen
results = model.predict(source="perros.jpg")

# Cargar la imagen original
image = cv2.imread("perros.jpg")

# Obtener todas las detecciones
detections = results[0]  # Primera imagen (si hay varias)

# Filtrar las detecciones solo para la clase "dog"
dog_class_id = [key for key, value in model.names.items() if value == 'dog'][0]

# Iterar sobre las detecciones para dibujar solo los perros
for box in detections.boxes:
    if int(box.cls) == dog_class_id:
        # Obtener las coordenadas del cuadro delimitador
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Dibujar el cuadro sobre la imagen original
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Verde para los perros
        # Agregar la etiqueta "dog"
        cv2.putText(image, "dog", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Mostrar la imagen con solo los perros detectados
cv2.imshow("Perros Detectados", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
