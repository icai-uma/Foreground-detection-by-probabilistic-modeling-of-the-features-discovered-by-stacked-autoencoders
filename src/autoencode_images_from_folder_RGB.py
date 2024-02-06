import numpy as np
import cv2
import os
import keras
import util
from sklearn.cluster import KMeans, DBSCAN

os.environ["CUDA_VISIBLE_DEVICES"]="0" #To force tensorflow to only see one GPU.

dataset_dir = "../../data/dynamicBackground/canoe/input"
encoder_model_path = "../models22/Ner_patches_TinyImage_autoencoder_model_RGB.h5py"
result_path = "../autoencoded"

mu, sigma = 0, 0.0 #Gaussian noise parameters.

patch_height = 16
patch_width = 16

original_image_height = 240
original_image_width = 320

if not os.path.exists(result_path):
    os.makedirs( result_path )

print ("Loading encoder model...")
autoencoder_model = keras.models.load_model(encoder_model_path)

print ("Loading data...")
dataset_dirs = os.listdir(dataset_dir)
dataset_dirs = sorted(dataset_dirs)
original_dataset_size = len(dataset_dirs)

original_dataset = np.zeros([original_dataset_size, original_image_height, original_image_width, 3])

index = 0
for img_name in dataset_dirs:
    img_dir = os.path.join(dataset_dir, img_name)
    img = cv2.imread( img_dir )
    original_dataset[index] = img.astype( float )
    index += 1

original_dataset /= 255.

original_dataset_with_noise = util.add_additive_gaussian_noise(original_dataset, mu, sigma)

print ("Transforming original data to patches...")
patched_dataset = util.split_in_patches_various_images(original_dataset_with_noise, patch_height, patch_width)

patched_dataset_reshaped = np.reshape(patched_dataset, [patched_dataset.shape[0], patch_height*patch_width*3])

print ("Autoencoding...")
predicted_patches = autoencoder_model.predict(patched_dataset_reshaped)
predicted_patches *= 255.
predicted_patches_reshaped = np.reshape(predicted_patches, [predicted_patches.shape[0], patch_height, patch_width, 3])

reconstructed_images = util.reconstruct_from_patches(predicted_patches_reshaped, original_image_height, original_image_width)

original_dataset_with_noise *= 255.
for ii in range(original_dataset_size):
    cv2.imwrite(os.path.join(result_path, str(ii) + ".png"), reconstructed_images[ii])
    cv2.imwrite(os.path.join(result_path, str(ii) + ".ori.png"), original_dataset_with_noise[ii])
