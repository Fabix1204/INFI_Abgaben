import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from PIL import Image

# Setze die Parameter
num_classes = 3
input_shape = (180, 180, 3)
batch_size = 32
epochs = 20
learning_rate = 0.001

# Lade die Daten
d = keras.preprocessing.image_dataset_from_directory('Aufgabe_06/img/resized/rps/rps/', image_size=(180, 180), label_mode='categorical', batch_size= 1000)

images = None
labels = None

print("Class names: %s" % d.class_names) # Welche Kategorien gibt es generell

for d, l in d.take(1):
    images = d
    labels = l

print(images.shape)
print(labels.shape)

# Erstelle das Modell
model = keras.Sequential(
    [ keras.Input(shape=input_shape),
     layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
     layers.MaxPooling2D(pool_size=(2, 2)),
     layers.Flatten(), layers.Dropout(0.5),
     layers.Dense(num_classes, activation="softmax"),
])

model.summary()

# ## Train the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 64
epochs = 10
validation_split = 0.1

history = model.fit(images, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

# Lade und verarbeite das Bild
image = Image.open('Aufgabe_06/img/resized/rps-validation/paper2.png')
image = image.resize((180, 180))
image = image.convert('RGB')  # Konvertiere das Bild zu RGB
image = np.array(image)

# Überprüfe die Form des Bildes
print(image.shape)

# Normalisiere das Bild und erweitere die Dimensionen
image = image.astype('float32') / 255
image = np.expand_dims(image, axis=0)

# Überprüfe die Form des Bildes nach der Erweiterung
print(image.shape)

# Mache eine Vorhersage
pred = model.predict(image)
print(pred)

# Speichere das Modell
model.save("Aufgabe_06/model/model.keras")
model.save_weights("Aufgabe_06/model/model.weights.h5")