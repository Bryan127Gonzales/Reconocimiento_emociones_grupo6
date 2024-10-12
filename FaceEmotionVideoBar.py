# Import de librerias
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt

# Definición de las emociones y colores para el gráfico
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
my_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#000000', '#FF00FF', '#00FFFF']

# Variables para calcular FPS
time_actualframe = 0
time_prevframe = 0

# Cargamos el modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el modelo de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Crear la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Inicializar gráfico
plt.ion()
x = list(range(len(classes)))  # Índices de las emociones
y = [0.0] * len(classes)  # Valores iniciales (0 para todas las emociones)
figura1, ax = plt.subplots()

def update_bar_chart(y):
    """Actualiza el gráfico de barras en tiempo real."""
    ax.clear()  # Limpia el gráfico antes de redibujarlo
    ax.bar(x, y, color=my_colors, width=0.8)  # Dibuja las barras
    ax.set_xticks(x)  # Posición de las etiquetas en el eje x
    ax.set_xticklabels(classes)  # Etiquetas de las emociones
    ax.set_ylim([0.0, 1.0])  # Escala del eje y (valores entre 0 y 1)
    ax.grid(True)
    plt.draw()  # Dibuja el gráfico
    plt.pause(0.01)  # Pausa breve para actualizar el gráfico

def predict_emotion(frame, faceNet, emotionModel):
    """Predice emociones en el frame recibido."""
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

while True:
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)

    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)

    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        (angry, disgust, fear, happy, neutral, sad, surprise) = pred

        label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(pred) * 100)
        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        # Actualiza el gráfico con los valores de predicción actuales
        y = [angry, disgust, fear, happy, neutral, sad, surprise]
        update_bar_chart(y)

    time_actualframe = time.time()
    if time_actualframe > time_prevframe:
        fps = 1 / (time_actualframe - time_prevframe)
    time_prevframe = time_actualframe

    cv2.putText(frame, str(int(fps))+" FPS", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
