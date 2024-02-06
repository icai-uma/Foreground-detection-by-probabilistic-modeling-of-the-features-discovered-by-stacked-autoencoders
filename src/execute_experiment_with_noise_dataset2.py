import time
import datetime
import numpy as np
import os
import keras
import tensorflow as tf
from keras import backend as k

initial_total_experiment_time = time.time()

###################################
os.environ["CUDA_VISIBLE_DEVICES"]="0" #To force tensorflow to only see one GPU.
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
###################################

encode_training_dataset = "../../data/tiny_images/tinyOutput3/train"

noises = ["0", "0.1", "0.2", "0.31622776601683794", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression10", "compression5", "compression1"]
dataset_name_list = ["canoe", "pedestrians", "port_0_17fps", "overpass", "boats", "fountain01", "fountain02"]

ALPHA_list = [0.001, 0.005, 0.01, 0.05]

version_list = ["version1.3", "version2.3", "version6.3", "version7.3"]

#Noise to add to training
gaussian_noise_added_to_training_mu = 0
gaussian_noise_added_to_training_sigma = 0

#Noise to add to all dataset.
image_gaussian_noise_mu = 0
image_gaussian_noise_sigma = 0

#Debug parameters.
debugOutputFile = "debug.out"
debug = False
print_list = []

#Output Video
output_video = False

#Do we generate BW images?
segmented_output = True

input_subfolder = False

input_subfolder_list = ["compression10", "compression5", "compression1"]

for version in version_list:

    if (version == "version1.3"):
        encoder_model_version = "models22"
    if (version == "version2.3"):
        encoder_model_version = "models24"
    if (version == "version6.3"):
        encoder_model_version = "models26"
    if (version == "version7.3"):
        encoder_model_version = "models27"
        
    encoder_model_path = "../network_models/" + encoder_model_version + "/Ner_patches_TinyImage_encoder_model_RGB.h5py"

    print ("Loading encoder model " + encoder_model_version)
    encoder_model = keras.models.load_model(encoder_model_path)

    for dataset_name in dataset_name_list:
            
        for noise in noises:
        
            if noise in input_subfolder_list:
                input_subfolder = True
            else:
                input_subfolder = False

            datasets_folder = "../datasets_with_noise/" + str(noise)

            for ALPHA in ALPHA_list:

	            if (dataset_name == "level_crossing"):
		            N = 150 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = None
		            dataset_dir = os.path.join(datasets_folder, dataset_name)
	            if (dataset_name == "water_surface"):
		            N = 30 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = None
		            dataset_dir = os.path.join(datasets_folder, dataset_name)
	            if (dataset_name == "highway"):
		            N = 469 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "baseline"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "office"):
		            N = 569 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "baseline"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "pedestrians"):
		            N = 299 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "baseline"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "PETS2006"):
		            N = 299 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "baseline"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "canoe"):
		            N = 799 #Number of images to obtain initial gaussian models.
		            print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
		            dataset_subfolder = "dynamicBackground"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "boats"):
		            N = 1899 #Number of images to obtain initial gaussian models.
		            print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
		            dataset_subfolder = "dynamicBackground"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "port_0_17fps"):
		            N = 999 #Number of images to obtain initial gaussian models.
		            #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
		            dataset_subfolder = "lowFramerate"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "overpass"):
		            N = 999 #Number of images to obtain initial gaussian models.
		            print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
		            dataset_subfolder = "dynamicBackground"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "streetCornerAtNight"):
		            N = 799 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "nightVideos"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "tramStation"):
		            N = 499 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "nightVideos"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "blizzard"):
		            N = 899 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "badWeather"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "wetSnow"):
		            N = 499 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "badWeather"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "fountain01"):
		            N = 399 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "dynamicBackground"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
	            if (dataset_name == "fountain02"):
		            N = 499 #Number of images to obtain initial gaussian models.
		            dataset_subfolder = "dynamicBackground"
		            dataset_dir = os.path.join(datasets_folder, dataset_subfolder, dataset_name)
		            
	            if (input_subfolder):
		            dataset_dir = os.path.join(dataset_dir, "input")
                    
	            result_path = os.path.join("../segmentationResult/" + version + "/ours", str(noise), dataset_subfolder, dataset_name)
	            result_path = os.path.join(result_path, "segmentation")

	            initial_segmentation_time = time.time()
	            print ("--------------------------------------------------------------------------------")		
	            print ("Starting Segmentation over dataset " + dataset_name + " with version " + version + " and parameter ALPHA = " + str(ALPHA))
	            execfile("segmentation_" + version + ".py")
	            total_experiment_time = time.time() - initial_segmentation_time
	            print ("Total experiment time: " + str(total_experiment_time))

    final_total_experiment_time = time.time()

    print("Total experiments time: " + str(datetime.timedelta(seconds=(final_total_experiment_time-initial_total_experiment_time))))

