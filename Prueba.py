import os
import pywt
import cv2 as cv
import numpy as np
from numpy.linalg import norm
from datasets import load_dataset
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)

# Nombre de clases
class_names = [
    'ANULAR_DER',
    'ANULAR_IZQ',
    'CORAZON_DER',
    'CORAZON_IZQ',
    'INDICE_DER',
    'INDICE_IZQ',
    'MEÑIQUE_DER',
    'MEÑIQUE_IZQ',
    'PULGAR_DER',
    'PULGAR_IZQ'
]

# Cargar datasets
datasets = [load_dataset(name) for name in class_names]

# Convertir imágenes a escala de grises
images = [[cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset['train']] for dataset in datasets]

# Función para extraer características
def extract_features(img):
    # Redimensionar para consistencia
    img = cv.resize(img, (128, 128))

    # Wavelet
    coeffs = pywt.dwt2(img, 'db1')
    cA, (cH, cV, cD) = coeffs
    wavelet_features = cA.flatten()

    # Gabor
    g_kernel = cv.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv.CV_32F)
    gabor_response = cv.filter2D(img, cv.CV_8UC3, g_kernel)
    gabor_features = gabor_response.flatten()

    # Fourier
    f_transform = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    fourier_features = magnitude_spectrum.flatten()

    # Concatenar todo
    features = np.concatenate([wavelet_features, gabor_features, fourier_features])
    return features

# Crear base de entrenamiento (primeras 300 imágenes de cada clase)
train_features = []
train_labels = []

for label_index, img_class in enumerate(images):
    for img in img_class[:300]:
        feats = extract_features(img)
        train_features.append(feats)
        train_labels.append(class_names[label_index])

train_features = np.array(train_features)

# Cargar imágenes de prueba desde la carpeta IMG_PRUEBAS
test_folder = 'IMG_PRUEBAS'
# Cargar datasets
img_pruebas = load_dataset(test_folder)
test_images = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in img_pruebas['train']]

# Convertir imágenes a escala de grises
images = [[cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset['train']] for dataset in datasets]

# Mostrar lista de imágenes disponibles
print("\nIMÁGENES DISPONIBLES EN IMG_PRUEBAS:")
for idx, name in enumerate(test_images):
    print(f"{idx}: {name}")

# Selección de imagen por índice
img_idx = int(input("\nIngrese el índice de la imagen que desea clasificar: "))
test_img = test_images[img_idx]
test_img_name = test_images[img_idx]

# Mostrar imagen seleccionada
plt.imshow(test_img, cmap='gray')
plt.title(f"Imagen seleccionada: {test_img_name}")
plt.axis('off')
plt.show()

# Clasificación
test_feats = extract_features(test_img)
dists = [norm(test_feats - train_feat) for train_feat in train_features]
min_idx = np.argmin(dists)
predicted_label = train_labels[min_idx]

# Resultado
print(f"\nRESULTADO:")
print(f"La imagen '{test_img_name}' se clasifica como: {predicted_label}")
