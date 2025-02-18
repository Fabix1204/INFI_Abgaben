from PIL import Image
import os
import numpy as np

folders = ['schere', 'papier', 'stein']
output_base_folder = 'Aufgabe_06/img/resized/'

# Erstelle den Basis-Ausgabeordner, falls er nicht existiert
if not os.path.exists(output_base_folder):
    os.makedirs(output_base_folder)

for folder in folders:
    folder_path = f'Aufgabe_06/img/{folder}/'
    output_folder = os.path.join(output_base_folder, folder)

    # Erstelle den spezifischen Ausgabeordner für jede Kategorie
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = np.array(Image.open(img_path).resize((180, 180)))
            output_path = os.path.join(output_folder, filename)  # Speichern im jeweiligen Ordner
            Image.fromarray(img).save(output_path)
