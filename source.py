import cv2 as cv
import numpy as np
from numpy.linalg import norm
from datasets import load_dataset

np.set_printoptions(threshold = np.inf)

#Paths carpetas datasets
datasets_path = [
        'ANULAR_DER',
        'ANULAR_IZQ',
        'CORAZON_DER', 
        'CORAZON_IZQ', 
        'INDICE_DER', 
        'INDICE_IZQ',
        'MEÑIQUE_DER',
        'MEÑIQUE_IZQ',
        'PULGAR_DER',
        'PULGAR_IZQ',]

#Carga de datasets
dataset_anular_der = load_dataset(datasets_path[0])
dataset_anular_izq = load_dataset(datasets_path[1])
dataset_corazon_der = load_dataset(datasets_path[2])
dataset_corazon_izq = load_dataset(datasets_path[3])
dataset_indice_der = load_dataset(datasets_path[4])
dataset_indice_izq = load_dataset(datasets_path[5])
dataset_meñique_der = load_dataset(datasets_path[6])
dataset_meñique_izq = load_dataset(datasets_path[7])
dataset_pulgar_der = load_dataset(datasets_path[8])
dataset_pulgar_izq = load_dataset(datasets_path[9])

#Guardar imagenes en array
anular_der_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_anular_der['train']]
anular_izq_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_anular_izq['train']]
corazon_der_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_corazon_der['train']]
corazon_izq_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_corazon_izq['train']]
indice_der_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_indice_der['train']]
indice_izq_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_indice_izq['train']]
meñique_der_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_meñique_der['train']]
meñique_izq_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_meñique_izq['train']]
pulgar_der_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_pulgar_der['train']]
pulgar_izq_imgs = [cv.cvtColor(np.array(sample['image']), cv.COLOR_RGB2GRAY) for sample in dataset_pulgar_izq['train']]

cv.imshow("imagen test", anular_der_imgs[0])  # Mostrar el nombre de la imagen
cv.waitKey(0)
cv.destroyAllWindows()