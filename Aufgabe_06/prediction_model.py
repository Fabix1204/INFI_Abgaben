import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import time

# Lade das Modell
model = keras.models.load_model("Aufgabe_06/model/model.keras")
model.load_weights("Aufgabe_06/model/model.weights.h5")

# Lade die Klassennamen
d = keras.preprocessing.image_dataset_from_directory('Aufgabe_06/img/resized/rps/rps/', image_size=(180, 180), label_mode='categorical', batch_size=1000)
class_names = d.class_names
print("Class names: ", class_names)

# Kamera initialisieren
cap = cv2.VideoCapture(0)

last_prediction_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Bild für das Modell vorbereiten
    img = cv2.resize(frame, (180, 180))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Vorhersage treffen und in der Konsole ausgeben, wenn eine Sekunde vergangen ist
    current_time = time.time()
    if current_time - last_prediction_time >= 1:
        predictions = model.predict(img)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        print(f"Prediction: {predicted_class} ({confidence:.2f})")
        last_prediction_time = current_time
    
    # Zeige das aktuelle Kamerabild an
    cv2.imshow("Live Camera", frame)
    
    # Beenden mit 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()