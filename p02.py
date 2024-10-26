from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# results = model("monos.jpg")
results = model("perros2.jpg")

for result in results:
    result.show()

# Aún no sé cómo pedirle que busque sólo una cosa, e indicarle el porcentaje.
# otra duda que tengo, es ¿la resolución de la imagen influye?

