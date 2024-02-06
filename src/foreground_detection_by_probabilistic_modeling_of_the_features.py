#Command to test version 1: python3.6 background_modeling_for_semantic_segmentation.py -m coco -d "../../data/changeDetection/*" -o "../output/test1"
#Command to test version 2: python3.6 background_modeling_for_semantic_segmentation.py -m coco -d "../../data/changeDetection/*" -o "../output/test2"
#Command to test version 2 on local: python3.6 background_modeling_for_semantic_segmentation.py -m coco -d "../../data/dataset2014/dataset/*" -o "../output/test2_C"

import os
import sys
import random
import math
import cv2
import argparse
import numpy as np
from scipy import misc, ndimage
from keras import backend as K
import keras
import tensorflow as tf
from glob import glob
import time

# Root directory
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
import Background_model
import my_utils                             # pylint: disable=import-error

folder_to_search = ["pedestrians", "baseline", "fountain01", "fountain02", "canoe", "boats", "dynamicBackground", "lowFramerate", "port_0_17fps"]                                                                                 # Number of images for train

def segmentation(images_path, output_path_in, encoder, category, dataset, n_img_to_train):
                                                                                                    # Code to execute semantic segmentation.                  
    output_path = os.path.join(output_path_in, category, dataset)                                   # We generate output path.
    if not os.path.isdir(output_path):                                                              # We check if output path exists.
        os.makedirs(output_path)                                                                    # We create path if it does not exist.

    total_image_number = len(images_path)                                                           # We get total number of images path.
    segmented_images_count = 0                                                                      # We set segmented images counter to 0.

    images_path = sorted(images_path)                                                               # We ensure images are ordered by name.
    for i, img_path in enumerate(images_path):                                                      # For each image path...
        if (os.path.isfile(img_path) and (img_path[-4:]==".png" or img_path[-4:]==".jpg")):         # If it is a file and has image extensión...
            segmented_images_count += 1                                                             # We update counter.
            print(("Processing image " + img_path + "\nProcessed {} / {} from " 
                + category + "/" + dataset).format(segmented_images_count,total_image_number))      # Print progress info.
            img = cv2.imread(img_path)                                                              # We load the image.    
            
            output_filename = os.path.join(output_path, 
                my_utils.generate_file_name_by_number(segmented_images_count))                      # We generate the segmented image path.

            if segmented_images_count == 1:                                                         # If this is the first image...
                back_model = Background_model.Background_model(encoder=encoder, 
                narrowest_layer_size=16, padding_value = 8, patch_height = 16, 
                patch_width = 16, slicing_window_height = 8, slicing_window_width = 8, 
                number_of_images_for_training = n_img_to_train)                                     # We initialize the background model from the first image.

            back_model.next_image(img)                                                              # We process the image with background model.
            
            if segmented_images_count > n_img_to_train:
                segmented_img = back_model.get_last_segmented_image()                               # We get segmented image from background model.
                cv2.imwrite(output_filename, segmented_img)                                         # We save the segmented image.         

        else:                                                                                       # This path is not a file or the file is not an image...
            total_image_number -= 1                                                                 # We update total image number.
            print(("This element is not an image : " + img_path + "\nProcessed {} / {} from " 
                + category + "/" + dataset).format(segmented_images_count,total_image_number))      # Print progress info.
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='../models/models_16/TinyImage_encoder_model_RGB.h5py',
                        help='Encoder model path')
    parser.add_argument('-d', '--datasets_dir', type=str, default=None, 
                        help="Path to the input datasets folder.")
    parser.add_argument('-o', '--output_path', type=str, default='./output',
                        help='Path to output')
    parser.add_argument('--id', default="0")
    
    args = parser.parse_args()

    # Handle output args
    if args.datasets_dir:
        fn, ext = os.path.splitext(args.output_path)
        print("Output folder: " + fn)
        if ext:
            parser.error("output_path should be a folder for multiple file input")
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path)
    
    print ("Loading encoder model from " + args.model)
    encoder_model = keras.models.load_model(args.model)
        

    # Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    print(args)        
    if args.datasets_dir:
                                                                                                                    # We need to process each dataset...
        subfolders_in_folder = sorted(glob(args.datasets_dir))
        for subfolder_path in subfolders_in_folder:
            if os.path.isdir(subfolder_path) and os.path.basename(subfolder_path) in folder_to_search:              # If this element is an important folder...       
                elements_in_subfolder = sorted(glob(os.path.join(subfolder_path, "*")))                             # Get elements in subfolder...
                for element_path in elements_in_subfolder:                    
                    if os.path.isdir(element_path) and os.path.basename(element_path) in folder_to_search:          # If this element is a folder...
                        category_path = subfolder_path                                                              # We will consider subfolder as a category
                        _, category = os.path.split(category_path)
                        dataset_path = element_path                                                                 # and this element as a dataset.
                        _, dataset = os.path.split(dataset_path)
                        dataset_elements = sorted(glob(os.path.join(dataset_path, "*")))
                        if os.path.join(dataset_path, "input") in dataset_elements:                                 # If there is an input folder
                            images = sorted(glob(os.path.join(dataset_path, "input", "*")))                         # we will get images from input folder.
                        else:
                             images = dataset_elements                                                              # Else, we get images from that folder.
                        category, n_img_to_train = my_utils.obtain_information_about_dataset(dataset)
                        segmentation(images, args.output_path, encoder_model, category, dataset, n_img_to_train)    # Now, we process images.
                    else:
                        if not os.path.isdir(element_path):
                            raise("Error: " + dataset_path + " is not a folder and it should be a dataset.")        # Error.            
            else:
                if not os.path.isdir(subfolder_path):
                    raise("Error: " + subfolder_path + " is not a folder")                                          # Error. There is no folder.