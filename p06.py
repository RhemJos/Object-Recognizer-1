from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado
model = YOLO("yolo11n.pt")

# Realizar la predicción en una imagen
results = model.predict(source="persons.jpg")

# Cargar la imagen original
image = cv2.imread("persons.jpg")

# Obtener todas las detecciones
detections = results[0]  # Primera imagen (si hay varias)

# Filtrar las detecciones solo para la clase "dog"
person_class_id = [key for key, value in model.names.items() if value == 'person'][0]

# Iterar sobre las detecciones para dibujar solo los perros
for box in detections.boxes:
    if int(box.cls) == person_class_id:
        # Obtener las coordenadas del cuadro delimitador
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Obtener el porcentaje de confianza
        # confidence = float(box.conf[0]) * 100  # Convertir a porcentaje
        confidence = float(box.conf[0])
        # Dibujar el cuadro sobre la imagen original
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Verde para los perros
        # Agregar la etiqueta
        label = f"person {confidence:.2f}"  # Formato con dos decimales
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

# Mostrar la imagen con solo los perros detectados
cv2.imshow("Personas Detectadas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
