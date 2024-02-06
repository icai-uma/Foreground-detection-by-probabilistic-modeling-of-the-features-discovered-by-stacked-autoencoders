import numpy as np
import cv2
import os
import keras
import util
from sklearn.cluster import KMeans, DBSCAN
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" #To force tensorflow to only see one GPU.

dataset_dir = "../../data/water_surface/all"
encoder_model_path = "../models4/Ner_patches_TinyImage_encoder_model_RGB.h5py"
result_path = "../results4/encoded_RGB"
patch_height = 16
patch_width = 16
original_image_height = 128
original_image_width = 160

print ("Loading encoder model...")
encoder_model = keras.models.load_model(encoder_model_path)

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

print ("Transforming original data to patches...")
patched_dataset = util.split_in_patches_various_images(original_dataset, patch_height, patch_width)

patched_dataset_reshaped = np.reshape(patched_dataset, [patched_dataset.shape[0], patch_height*patch_width*3])

print ("Encoding...")
predicted_patches = encoder_model.predict(patched_dataset_reshaped)
predicted_patches_reshaped = np.reshape(predicted_patches, [predicted_patches.shape[0], 1, 1, 64])

reconstructed_images = util.reconstruct_from_patches(predicted_patches_reshaped, original_image_height/patch_height, original_image_width/patch_width)

print reconstructed_images

util.to_output_array_file(reconstructed_images, result_path + str("/output_matrix"), "matlab")
