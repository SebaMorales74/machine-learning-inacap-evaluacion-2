import cv2
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Definir el tamaño de las imágenes
image_size = image_width, image_height = 200, 200
data_folder = r'productos'

# 1. Preparación de datos para entrenamiento con aumento de datos más agresivo
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización de las imágenes
    rotation_range=40,  # Más rotación
    zoom_range=0.3,  # Más zoom
    width_shift_range=0.2,  # Más desplazamiento horizontal
    height_shift_range=0.2,  # Más desplazamiento vertical
    horizontal_flip=True,  # Mantener el flip horizontal
    vertical_flip=True,  # Añadir flip vertical
    brightness_range=[0.7, 1.3],  # Más variación de brillo
    fill_mode='nearest'
)

# División de los datos en entrenamiento, validación y prueba
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización de las imágenes
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
    validation_split=0.3  # 70% entrenamiento, 30% validación/prueba
)

# Generador de datos de entrenamiento
train_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Generador de datos de validación
validation_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# 2. Construcción del modelo CNN mejorado
def construir_modelo():
    model = Sequential([
        Input(shape=(image_width, image_height, 3)),  # Tamaño de entrada
        Conv2D(32, (3, 3), activation='relu'),  # Capa convolucional
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3), activation='relu'),  # Añadir más filtros
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),  # Aumentar la densidad de la capa
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(5, activation='softmax')  # 5 clases: "bototos", "cascos", "guantes", "pantalones" y "reflectantes"
    ])

    # Reducir la tasa de aprendizaje
    model.compile(
        optimizer=Adam(learning_rate=0.00005),  # Tasa de aprendizaje más baja
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 3. Añadir Early Stopping para detener el entrenamiento si no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Crear el modelo
model = construir_modelo()

# Entrenar el modelo
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,  # Aumentar a 50 épocas para mayor precisión
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[early_stopping]  # Añadir Early Stopping
)

# Evaluar el modelo
loss, accuracy = model.evaluate(validation_generator)
print(f"Pérdida: {loss:.2f}")

# Obtener predicciones y etiquetas verdaderas
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = validation_generator.classes

# Generar matriz de confusión
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.show()

# Reporte de clasificación
print(classification_report(y_true, y_pred_classes, target_names=validation_generator.class_indices.keys()))

# Guardar el modelo entrenado
model.save('./modelo_reconocimiento_imagenes.h5')

# Crear el diccionario de etiquetas dinámicamente
etiquetas = {v: k for k, v in train_generator.class_indices.items()}

def predecir_imagen(img_array):
    # Redimensionar la imagen a 200x200 (el tamaño esperado por el modelo)
    img_array = cv2.resize(img_array, image_size)  
    img_array = np.expand_dims(img_array, axis=0)  # Agregar dimensión para representar el batch
    img_array = img_array / 255.0  # Normalizar la imagen
    
    # Realizar la predicción
    prediccion = model.predict(img_array)
    
    # Devolver la clase predicha y la predicción completa
    return np.argmax(prediccion), prediccion

# Captura de video en tiempo real
cap = cv2.VideoCapture(0)

# Modificar la sección de captura de video
while True:
    ret, frame = cap.read()  # Capturar el frame de la cámara
    if not ret:
        print("Error: No se pudo capturar el frame")
        break

    # Dibujar un cuadro de alineación en el centro del frame
    height, width, _ = frame.shape
    x1, y1 = int(width * 0.3), int(height * 0.3)
    x2, y2 = int(width * 0.7), int(height * 0.7)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Camara', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        roi = frame[y1:y2, x1:x2]  # Región de interés (ROI)
        clase_predicha, prediccion = predecir_imagen(roi)

        confianza = np.max(prediccion)
        umbral_confianza = 0.20

        # Mostrar todas las probabilidades con sus etiquetas
        for i, prob in enumerate(prediccion[0]):
            print(f"{etiquetas[i]}: {prob:.2f}")

        # Mostrar la clase con la probabilidad más alta
        clase_nombre = etiquetas[clase_predicha]
        print(f"\nClase con mayor probabilidad: {clase_nombre} ({confianza:.2f})")

        if confianza < umbral_confianza:
            print(f"\nConfianza demasiado baja ({confianza:.2f}).")

    elif key == ord('q'):
        print("Saliendo del programa...")
        break

cap.release()
cv2.destroyAllWindows()