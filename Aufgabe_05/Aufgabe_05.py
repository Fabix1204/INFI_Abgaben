from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import collections
import matplotlib.pyplot as plt
import pandas as pd
import json
import keras
from keras import layers
from PIL import Image


"""
## Prepare the data
"""

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

"""
 the data, split between train and test sets
x_train: images for training
y_train: labels for training
x_test: images for testing the model
y_test: labels for testing the model
"""
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
#download the images
"""
path = "C:/Users/fabia/OneDrive - HTL Anichstrasse/Dokumente/HTL/5AHWII/INFI/Aufgaben/INFI_Abgaben/Aufgabe_05/img/"

print ("storing images.....")
for i in range (20):
    img = Image.fromarray(x_train[i])
    img.save(path + str(i) + ".png")



"""
# Scale images to the [0, 1] range
# Cast to float values before to make sure result ist float
"""
x_train = x_train.astype("float32") / 255
print(x_train.shape, "train samples")
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")

nr_labels_y = collections.Counter(y_train) #count the number of labels
print(nr_labels_y, "Number of labels")

# convert class vectors (the labels) to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_labels = y_test #use this to leave the labels untouched
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
## Build the model
"""

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = keras.Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

"""
## Train the model
"""

batch_size = 64
epochs = 8

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback])

#draw the learn function
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

"""
How to load and save the model
"""

model.save('C:/Users/fabia/OneDrive - HTL Anichstrasse/Dokumente/HTL/5AHWII/INFI/Aufgaben/INFI_Abgaben/Aufgabe_05/model/model.h5')
model.save_weights("C:/Users/fabia/OneDrive - HTL Anichstrasse/Dokumente/HTL/5AHWII/INFI/Aufgaben/INFI_Abgaben/Aufgabe_05/model/model.weights.h5")

weights = model.get_weights()
j =json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)
print(j)

model = keras.models.load_model('C:/Users/fabia/OneDrive - HTL Anichstrasse/Dokumente/HTL/5AHWII/INFI/Aufgaben/INFI_Abgaben/Aufgabe_05/model/model.h5')
model.load_weights("C:/Users/fabia/OneDrive - HTL Anichstrasse/Dokumente/HTL/5AHWII/INFI/Aufgaben/INFI_Abgaben/Aufgabe_05/model/model.weights.h5")

model.summary()

model_json = model.to_json()

score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Predictions

pred = model.predict(x_test)
print(pred[2])
print(y_labels[2])
pred_i = np.argmax(pred[2])
print(pred_i)
