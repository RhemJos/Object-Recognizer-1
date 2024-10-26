from ultralytics import YOLO
import cv2

# Cargar el modelo preentrenado
model = YOLO("yolo11n.pt")  # Cambia por otro modelo si lo deseas

# Umbral de confianza
confidence_threshold = 0.45
# Archivo
archivo = "cars.jpg"
# Word
word = 'car'
word_spanish = "Carros"
# Contador de objetos que cumplen con el umbral de confianza
count = 0

# Realizar la predicción en una imagen
results = model.predict(source=archivo)

# Cargar la imagen original
image = cv2.imread(archivo)

# Obtener todas las detecciones
detections = results[0]  # Primera imagen (si hay varias)

# Filtrar las detecciones solo para la clase "dog"
class_id = [key for key, value in model.names.items() if value == word][0]

# Iterar sobre las detecciones para dibujar solo los perros con un porcentaje de confianza superior al umbral
for box in detections.boxes:
    # Obtener el porcentaje de confianza
    confidence = float(box.conf[0])

    # Filtrar según el umbral de confianza
    if int(box.cls) == class_id and confidence >= confidence_threshold:
        # Aumentar el contador de objetos detectados
        count += 1
        # Obtener las coordenadas del cuadro delimitador
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Dibujar el cuadro sobre la imagen original
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Verde para los perros

        # Agregar la etiqueta "dog" y la confianza
        label = f"{word} {confidence:.2f}"  # Formato con dos decimales
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Mostrar la imagen con solo los perros detectados que cumplan con el umbral de confianza
cv2.imshow(f"{word_spanish} Detectados con Confianza >= Umbral", image)
# Mostrar la cantidad de perros detectados que cumplen con el umbral
print(f"Cantidad de objetos detectados con confianza >= {confidence_threshold}: {count}")

cv2.waitKey(0)
cv2.destroyAllWindows()
