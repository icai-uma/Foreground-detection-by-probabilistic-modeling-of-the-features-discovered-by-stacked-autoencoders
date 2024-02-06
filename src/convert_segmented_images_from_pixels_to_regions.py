import os
import util
import cv2

dataset_source_folder
objective_dataset_folder
patch_height = 8
patch_width = 8
minimum_percentage_to_foreground = 50

dataset_list = ["level_crossing", "water_surface", "highway", "office", "pedestrians", "PETS2006", "canoe", "boats", "port_0_17fps", "overpass", "streetCornerAtNight", "tramStation", "blizzard", "wetSnow", "fountain01", "fountain02"]

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

    img_dirs = os.listdir(dataset_dir)
    img_dirs = sorted(img_dirs)

    for img_name in img_dirs:
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread(img_dir)

        new_img = util.convert_foreground_image_matrix_from_pixel_to_region(img, patch_height, patch_width, minimum_percentage_to_foreground)

        new_img_dir = os.path.join(objective_dataset_folder, img_name)
        cv2.imwrite(new_img, new_img_dir)