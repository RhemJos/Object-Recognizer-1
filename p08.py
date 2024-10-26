from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado
model = YOLO("yolo11n.pt")  # Cambia por otro modelo si lo deseas

# Umbral de confianza (ajústalo según tu necesidad)
confidence_threshold = 0.1  # En porcentaje, ejemplo: 50% como mínimo

# Realizar la predicción en una imagen
results = model.predict(source="cars.jpg")

# Cargar la imagen original
image = cv2.imread("cars.jpg")

# Obtener todas las detecciones
detections = results[0]  # Primera imagen (si hay varias)

# Filtrar las detecciones solo para la clase "dog"
car_class_id = [key for key, value in model.names.items() if value == 'car'][0]

# Iterar sobre las detecciones para dibujar solo los perros con un porcentaje de confianza superior al umbral
for box in detections.boxes:
    # Obtener el porcentaje de confianza
    confidence = float(box.conf[0])

    # Filtrar según el umbral de confianza
    if int(box.cls) == car_class_id and confidence >= confidence_threshold:
        # Obtener las coordenadas del cuadro delimitador
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dibujar el cuadro sobre la imagen original
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Verde para los perros

        # Agregar la etiqueta "dog" y la confianza
        label = f"car {confidence:.2f}"  # Formato con dos decimales
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Mostrar la imagen con solo los perros detectados que cumplan con el umbral de confianza
cv2.imshow("Carros Detectados con Confianza >= Umbral", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
