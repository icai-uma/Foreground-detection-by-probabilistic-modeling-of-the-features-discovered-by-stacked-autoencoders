import time
import datetime
import numpy as np
import os
import keras
import tensorflow as tf
from keras import backend as k

initial_total_experiment_time = time.time()

encoder_model_version = "models22"
encoder_model_path = "../network_models/" + encoder_model_version + "/Ner_patches_TinyImage_encoder_model_RGB.h5py"

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


print ("Loading encoder model.")
encoder_model = keras.models.load_model(encoder_model_path)

noise_sigma_videos = ["uniform","mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper"]
for noise_sigma_video in noise_sigma_videos:

    datasets_folder = "../datasets_with_noise/" + str(noise_sigma_video)

    #dataset_name = "boats"

    # ALPHA = 0.001 #Learning rate.
    # K = 7 #Threshold.
    # C = 15 #Component threshold. If there are more than C components classified as foreground, the patch will be classified as foreground.

    #Noise to add to training
    gaussian_noise_added_to_training_mu = 0
    gaussian_noise_added_to_training_sigma = 0.05

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

    dataset_name_list = ["canoe", "pedestrians", "port_0_17fps", "overpass", "boats", "fountain01","fountain02"]
    #dataset_name_list = ["streetCornerAtNight", "tramStation", "blizzard", "wetSnow"]
    #dataset_name_list = ["blizzard", "wetSnow"]

    K_list = [2, 3, 4, 5, 6, 7, 8]
    #K_list = [5, 6, 7]
    K_list = [7,8]

    C_list = [3, 6, 9, 12, 15]
    #C_list = [3, 6]

    ALPHA_list = [0.001, 0.005, 0.01, 0.05] #0.05

    configurations_lists = [["overpass", 6, 3, 0.001], ["overpass", 5, 6, 0.001], ["overpass", 4, 9, 0.001], 
    ["pedestrians", 7, 15, 0.001], ["pedestrians", 7, 12, 0.001], ["pedestrians", 6, 15, 0.001], 
    ["canoe", 5, 3, 0.001], ["canoe", 4, 6, 0.001], ["canoe", 3, 12, 0.001], 
    ["port_0_17fps", 5, 15, 0.05], ["port_0_17fps", 5, 12, 0.05], ["port_0_17fps", 5, 12, 0.01], 
    ["boats", 4, 3, 0.001], ["boats", 3, 9, 0.001],  ["boats", 3, 6, 0.001],  
    ["fountain01", 7, 12, 0.001], ["fountain01", 6, 15, 0.001], ["fountain01", 6, 12, 0.001],
    ["fountain02", 6, 15, 0.001], ["fountain02", 5, 15, 0.001], ["fountain02", 5, 12, 0.001]]

    for [dataset_name, K, C, ALPHA] in configurations_lists:

	    if (dataset_name == "level_crossing"):
		    N = 150 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, dataset_name)
	    if (dataset_name == "water_surface"):
		    N = 30 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasetss_folder, dataset_name)
	    if (dataset_name == "highway"):
		    N = 469 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(dataset_folder, "baseline", dataset_name)
	    if (dataset_name == "office"):
		    N = 569 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "baseline", dataset_name)
	    if (dataset_name == "pedestrians"):
		    N = 299 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "baseline", dataset_name)
	    if (dataset_name == "PETS2006"):
		    N = 299 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "baseline", dataset_name)
	    if (dataset_name == "canoe"):
		    N = 799 #Number of images to obtain initial gaussian models.
		    print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
		    dataset_dir = os.path.join(datasets_folder, "dynamicBackground", dataset_name)
	    if (dataset_name == "boats"):
		    N = 1899 #Number of images to obtain initial gaussian models.
		    print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
		    dataset_dir = os.path.join(datasets_folder, "dynamicBackground", dataset_name)
	    if (dataset_name == "port_0_17fps"):
		    N = 999 #Number of images to obtain initial gaussian models.
		    #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
		    dataset_dir = os.path.join(datasets_folder, "lowFramerate", dataset_name)
	    if (dataset_name == "overpass"):
		    N = 999 #Number of images to obtain initial gaussian models.
		    print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
		    dataset_dir = os.path.join(datasets_folder, "dynamicBackground", dataset_name)
	    if (dataset_name == "streetCornerAtNight"):
		    N = 799 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "nightVideos", dataset_name)
	    if (dataset_name == "tramStation"):
		    N = 499 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "nightVideos", dataset_name)
	    if (dataset_name == "blizzard"):
		    N = 899 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "badWeather", dataset_name)
	    if (dataset_name == "wetSnow"):
		    N = 499 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "badWeather", dataset_name)
	    if (dataset_name == "fountain01"):
		    N = 399 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "dynamicBackground", dataset_name)
	    if (dataset_name == "fountain02"):
		    N = 499 #Number of images to obtain initial gaussian models.
		    dataset_dir = os.path.join(datasets_folder, "dynamicBackground", dataset_name)

	    result_path = os.path.join("../segmentationResult/version1/ours",str(noise_sigma_video), dataset_name)
	    result_path = os.path.join(result_path, "segmentation")

	    initial_segmentation_time = time.time()
	    print ("-------------------------------------------------------------------------------------")		
	    print ("Starting Segmentation over dataset " + dataset_name + " with parameters K = " + str(K) + " C = " + str(C) + " ALPHA = " + str(ALPHA))
	    execfile("segmentation_version1.py")
	    total_experiment_time = time.time() - initial_segmentation_time
	    print ("Total experiment time: " + str(total_experiment_time))

    final_total_experiment_time = time.time()

    print("Total experiments time: " + str(datetime.timedelta(seconds=(final_total_experiment_time-initial_total_experiment_time))))

