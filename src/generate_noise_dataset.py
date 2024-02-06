import os
import numpy as np
import noise_util as n_util


dataset_source_folder = "../../data"
dataset_objective_folder = "../datasets_with_noise/compression5"

noise_mu, noise_sigma = 0, 0.2

#dataset_list = ["level_crossing", "water_surface", "highway", "office", "pedestrians", "PETS2006", "canoe", "boats", "port_0_17fps", "overpass", "streetCornerAtNight", "tramStation", "blizzard", "wetSnow", "fountain01", "fountain02"]
dataset_list = ["pedestrians", "canoe", "boats", "port_0_17fps", "fountain01", "fountain02", "overpass"]

for dataset_name in dataset_list:
	if (dataset_name == "level_crossing"):
	    N = 150 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "all")
	    dataset_subfolder = ""
	if (dataset_name == "water_surface"):
	    N = 30 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "all")
	    dataset_subfolder = ""
	if (dataset_name == "highway"):
	    N = 469 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "baseline"
	if (dataset_name == "office"):
	    N = 569 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "baseline"
	if (dataset_name == "pedestrians"):
	    N = 299 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "baseline"
	if (dataset_name == "PETS2006"):
	    N = 299 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "baseline", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "baseline"
	if (dataset_name == "canoe"):
	    N = 799 #Number of images to obtain initial gaussian models.
	    print_list = [[2,20], [9,20], [11,20], [15,20], [18,20], [24,20]]
	    dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "dynamicBackground"
	if (dataset_name == "boats"):
	    N = 1899 #Number of images to obtain initial gaussian models.
	    print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
	    dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "dynamicBackground"
	if (dataset_name == "port_0_17fps"):
	    N = 999 #Number of images to obtain initial gaussian models.
	    #print_list = [[4,5], [5,20], [11,36], [19,32], [18,13], [28,13]]
	    dataset_dir = os.path.join(dataset_source_folder, "lowFramerate", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "lowFramerate"
	if (dataset_name == "overpass"):
	    N = 999 #Number of images to obtain initial gaussian models.
	    print_list = [[16,23], [18,23], [20,23], [25,13], [25,20], [26,15]]
	    dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "dynamicBackground"
	if (dataset_name == "streetCornerAtNight"):
	    N = 799 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "nightVideos", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "nightVideos"
	if (dataset_name == "tramStation"):
	    N = 499 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "nightVideos", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "nightVideos"
	if (dataset_name == "blizzard"):
	    N = 899 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "badWeather", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "badWeather"
	if (dataset_name == "wetSnow"):
	    N = 499 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "badWeather", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "badWeather"
	if (dataset_name == "fountain01"):
	    N = 399 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "dynamicBackground"
	if (dataset_name == "fountain02"):
	    N = 499 #Number of images to obtain initial gaussian models.
	    dataset_dir = os.path.join(dataset_source_folder, "dynamicBackground", dataset_name)
	    dataset_dir = os.path.join(dataset_dir, "input")
	    dataset_subfolder = "dynamicBackground"

	#dataset_objective_dir = os.path.join(dataset_objective_folder, str(noise_sigma), dataset_subfolder, dataset_name)
	#util.generate_image_dataset_with_Gaussian_noise(dataset_dir, dataset_objective_dir, noise_mu, noise_sigma)
	dataset_objective_dir = os.path.join(dataset_objective_folder, dataset_subfolder, dataset_name)
	#util.generate_image_dataset_with_mask_noise_by_patches(dataset_dir, dataset_objective_dir, 0.2, 2)
	n_util.generate_image_dataset_with_compression_noise(dataset_dir, dataset_objective_dir, 5)
	#n_util.generate_image_dataset_with_uniform_noise(dataset_dir, dataset_objective_dir, low = -0.5, high = 0.5)
	#util.generate_image_dataset_with_salt_pepper_noise(dataset_dir, dataset_objective_dir, 0.2)
