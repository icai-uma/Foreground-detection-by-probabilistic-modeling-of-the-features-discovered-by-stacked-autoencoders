import cv2
import numpy as np
import os
from glob import glob

def generate_file_name_by_number(file_number, number_of_digits = 6, prefix = "bin", suffix = ".png"):
    file_name = "error"
    zeroes_to_add = number_of_digits - len(str(file_number))
    
    if zeroes_to_add < 0:
        raise("Error: number of digits lesser than file_number length.")
    else:
        file_name = prefix + zeroes_to_add * "0" + str(file_number) + suffix
        
    return file_name

def mixture_of_two_images(img1_dir, img2_dir, output_dir):
    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)

    new_img = np.zeros(img1.shape)
    new_img[:,:,0] = img1[:,:,0] # ours blue
    new_img[:,:,2] = img2[:,:,0] # other red

    cv2.imwrite(output_dir, new_img)

def mixture_of_images(imageset1_dir, imageset2_dir, output_dir):
    imageset1 = sorted(glob(os.path.join(imageset1_dir,"*")))
    imageset2 = sorted(glob(os.path.join(imageset2_dir,"*")))

    for img1_dir in imageset1:
        _, img_name = os.path.split(img1_dir)
        output_img_dir = os.path.join(output_dir, img_name)
        img2_dir = os.path.join(imageset2_dir, img_name)
        if img2_dir in imageset2:
            mixture_of_two_images(img1_dir, img2_dir, output_img_dir)

def obtain_information_about_dataset(dataset_name):

    if (dataset_name == "highway"):
        number_of_training_images = 469 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "office"):
        number_of_training_images = 569 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "pedestrians"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "PETS2006"):
        number_of_training_images = 299 #Number of images to obtain initial gaussian models.
        category ="baseline"
        
    if (dataset_name == "canoe"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
        category ="dynamicBackground"
        
    if (dataset_name == "boats"):
        number_of_training_images = 1899 #Number of images to obtain initial gaussian models.
        print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
        category ="dynamicBackground"
        
    if (dataset_name == "port_0_17fps"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
        category ="lowFramerate"
        
    if (dataset_name == "overpass"):
        number_of_training_images = 999 #Number of images to obtain initial gaussian models.
        print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
        category ="dynamicBackground"
        
    if (dataset_name == "streetCornerAtNight"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="nightVideos"
        
    if (dataset_name == "tramStation"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category = "nightVideos"
        
    if (dataset_name == "blizzard"):
        number_of_training_images = 899 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "wetSnow"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "skating"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "snowFall"):
        number_of_training_images = 799 #Number of images to obtain initial gaussian models.
        category ="badWeather"
        
    if (dataset_name == "fountain01"):
        number_of_training_images = 399 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
        
    if (dataset_name == "fountain02"):
        number_of_training_images = 499 #Number of images to obtain initial gaussian models.
        category ="dynamicBackground"
     
    return category, number_of_training_images
        
    
