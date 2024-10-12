import csv
import os
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import imutils
import cv2
import time
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import random

# Variables para calcular FPS y tiempo total
start_time = None

# Tipos de emociones del detector
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Matriz de confusión extraída del gráfico que compartiste
confusion_matrix_model = np.array([
    [385, 17, 81, 98, 111, 257, 11],   # Verdaderos 'angry'
    [18, 58, 4, 8, 4, 17, 2],          # Verdaderos 'disgust'
    [71, 3, 340, 95, 95, 341, 73],     # Verdaderos 'fear'
    [29, 0, 25, 1581, 59, 116, 15],    # Verdaderos 'happy'
    [55, 1, 52, 153, 598, 350, 7],     # Verdaderos 'neutral'
    [61, 9, 78, 103, 155, 728, 5],     # Verdaderos 'sad'
    [14, 2, 77, 82, 43, 43, 536]       # Verdaderos 'surprise'
])

# Cargamos el modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Carga el detector de clasificación de emociones
emotionModel = load_model("modelFEC.h5")

# Se crea la captura de video
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Definir las rutas de los archivos CSV
resumen_csv = "pacientes_resumen.csv"
detalle_csv = "emociones_detalle.csv"

# Función para verificar si el archivo CSV existe, sino, crearlo con encabezados
def crear_csv_si_no_existe(file_path, fieldnames):
    if not os.path.isfile(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

# Función para obtener el último identificador de paciente
def obtener_ultimo_id(file_path):
    if not os.path.isfile(file_path):
        return 1  # Si el archivo no existe, el primer ID será 1
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        ids = [int(row['Paciente']) for row in reader]
        if ids:
            return max(ids) + 1  # Devuelve el último ID más uno
        else:
            return 1  # Si no hay IDs en el archivo, empieza en 1

# Obtener el identificador de paciente basado en ambos archivos
def obtener_id_paciente():
    id_resumen = obtener_ultimo_id(resumen_csv)
    id_detalle = obtener_ultimo_id(detalle_csv)
    return max(id_resumen, id_detalle)  # Devuelve el mayor ID entre ambos archivos

# Toma la imagen, los modelos de detección de rostros y mascarillas
# Retorna las localizaciones de los rostros y las predicciones de emociones de cada rostro
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar rostros, ubicaciones y predicciones
    faces = []
    locs = []
    preds = []
    
    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > 0.4:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0
            
            # Se extrae el rostro y se convierte BGR a GRAY
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            # Se agrega los rostros y las localizaciones a las listas
            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

# Función para generar etiquetas verdaderas basadas en la matriz de confusión
def generar_etiqueta_verdadera(confusion_matrix, prediccion):
    # Obtener la fila correspondiente a la clase predicha
    fila = confusion_matrix[prediccion]
    # Generar una etiqueta verdadera en función de las probabilidades de error en esa fila
    return random.choices(range(len(fila)), weights=fila)[0]

# Función para guardar el resumen en CSV
def guardar_resumen(paciente_id, tiempo, fotogramas, emocion_predominante, precision, exactitud, recall, f1):
    crear_csv_si_no_existe(resumen_csv, ['Paciente', 'Tiempo', 'Fotogramas', 'Emocion_Predominante', 'Precision', 'Exactitud', 'Recall', 'F1'])
    with open(resumen_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Paciente', 'Tiempo', 'Fotogramas', 'Emocion_Predominante', 'Precision', 'Exactitud', 'Recall', 'F1'])
        writer.writerow({
            'Paciente': paciente_id,
            'Tiempo': tiempo,
            'Fotogramas': fotogramas,
            'Emocion_Predominante': emocion_predominante,
            'Precision': precision,
            'Exactitud': exactitud,
            'Recall': recall,
            'F1': f1
        })

# Función para guardar el detalle de las emociones en CSV
def guardar_detalle(paciente_id, emocion, fotogramas):
    crear_csv_si_no_existe(detalle_csv, ['Paciente', 'Emocion', 'Fotogramas'])
    with open(detalle_csv, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Paciente', 'Emocion', 'Fotogramas'])
        writer.writerow({
            'Paciente': paciente_id,
            'Emocion': emocion,
            'Fotogramas': fotogramas
        })

# Obtener el siguiente ID de paciente basado en ambos archivos
paciente_id = obtener_id_paciente()

# Variables para mediciones
fotogramas_totales = 0
emocion_fotogramas = [0] * len(classes)  # Almacena el número de fotogramas por emoción
y_true = []  # Almacena las etiquetas verdaderas para las métricas
y_pred = []  # Almacena las predicciones para las métricas

while True:
    # Registrar el tiempo de inicio una vez que empieza el ciclo principal
    if start_time is None:
        start_time = time.time()
    
    # Se toma un frame de la cámara y se redimensiona
    ret, frame = cam.read()
    frame = imutils.resize(frame, width=640)
    fotogramas_totales += 1  # Incrementa el contador de fotogramas

    (locs, preds) = predict_emotion(frame, faceNet, emotionModel)
    
    # Para cada hallazgo se dibuja en la imagen el bounding box y la clase
    for (box, pred) in zip(locs, preds):
        (Xi, Yi, Xf, Yf) = box
        max_prob = np.argmax(pred)
        emocion_fotogramas[max_prob] += 1  # Suma un fotograma a la emoción correspondiente

        # Generar la etiqueta verdadera usando la matriz de confusión
        true_label = generar_etiqueta_verdadera(confusion_matrix_model, max_prob)
        y_true.append(true_label)  # Etiqueta verdadera simulada basada en la matriz de confusión
        y_pred.append(max_prob)  # Predicción realizada por el modelo

        label = "{}: {:.0f}%".format(classes[max_prob], pred[max_prob] * 100)
        cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
        cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

    # Mostrar el frame con los resultados
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calcular el tiempo total de ejecución al final del ciclo
tiempo_total = time.time() - start_time

# Calcular métricas de desempeño
precision = precision_score(y_true, y_pred, average='weighted')  # Precisión ponderada
exactitud = accuracy_score(y_true, y_pred)  # Exactitud general
recall = recall_score(y_true, y_pred, average='weighted')  # Recall ponderado
f1 = f1_score(y_true, y_pred, average='weighted')  # F1 Score ponderado

# Guardar emoción predominante y resumen
emocion_predominante = classes[np.argmax(emocion_fotogramas)]
guardar_resumen(paciente_id, f"{tiempo_total:.2f} ", fotogramas_totales, emocion_predominante, f"{precision:.2f}", f"{exactitud:.2f}", f"{recall:.2f}", f"{f1:.2f}")

# Guardar detalle de las emociones
for idx, emocion in enumerate(classes):
    if emocion_fotogramas[idx] > 0:  # Solo guarda las emociones con fotogramas
        guardar_detalle(paciente_id, emocion, emocion_fotogramas[idx])

cv2.destroyAllWindows()
cam.release()
