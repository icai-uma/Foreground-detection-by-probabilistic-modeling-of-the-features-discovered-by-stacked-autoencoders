import os
import numpy as np
import cv2
import sys

def evaluate_dataset(path_GT, path_BW, prefix = None, sufix = None, add_zeros_before_number = False):

    if (path_GT[-1] != '/'):
        path_GT = path_GT + '/'
        
    if (path_BW[-1] != '/'):
        path_BW = path_BW + '/'
        
    info_files_GT = os.listdir(path_GT)
    info_files_GT = sorted(info_files_GT)
    info_files_BW = os.listdir(path_BW)
    info_files_BW = sorted(info_files_BW)
    medias = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    cont = 0
    numFrames = 0


    #We must get the temporal roi information if it exists.
    [gt_path_main_folder, _] = os.path.split(path_GT[:-1])
    temporal_roi_file = os.path.join(gt_path_main_folder, 'temporalROI.txt')
    if (os.path.isfile(temporal_roi_file)):
        #If it exists, we will use it as loop initiallization point.
        roi_info_file = open(temporal_roi_file, "r")
        roi_info = roi_info_file.readline()
        [first_element_in_roi, last_element_in_roi] = str.split(roi_info) #Info about the first and the last frame to study.
        first_element_in_roi = int(first_element_in_roi)
        last_element_in_roi = int(last_element_in_roi)
    else:
        first_element_in_roi = 1

    for i in range(len(info_files_GT)):
        if (os.path.isfile(os.path.join(path_GT, info_files_GT[i]))):
            #Is a file.
            filename_GT = info_files_GT[i]
            [_, name_ext] = os.path.split(filename_GT)
            [name, ext] = os.path.splitext(name_ext)
            image_number = int(name[2:])
            if (image_number >= first_element_in_roi):
                # We get those with .bmp, .jpg or .png extensions.
                if ((ext == '.bmp') | (ext == '.jpg') | (ext == '.png')):
                    path_img_GT = os.path.join(path_GT, filename_GT)
                    numFrames += 1
            
                    prefix_0s_length = len(name[2:]) - len(str(image_number))
                    zeros_string = ""
                    if (add_zeros_before_number):
                        for j in range(prefix_0s_length):
                            zeros_string = zeros_string + '0'
                    filename_bw = prefix + zeros_string + str(image_number) + sufix

                    path_img_BW = os.path.join(path_BW, filename_bw)
                    
                    
                    if(os.path.isfile(path_img_BW)): #If its segmented image does exist
                        #print(path_img_BW)
                        if(cont == 0):
                            medias = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])
                        else:
                            medias = np.vstack((medias, np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])))
                        medias[cont,0:10] = measureCDnet(path_img_BW, path_img_GT)
                        medias[cont,10] = i
                        cont = cont + 1
    
    eps = sys.float_info.epsilon
    media = np.sum(medias[:,0:8], 0)/cont
    TP = medias[:,8]
    FN = medias[:,9]
    foreground_pixels = TP + FN
    #We should recalculate median for some measures.
    #S should only count frames with foreground
    media[0] = np.sum(medias[:,0], 0)/(np.sum(foreground_pixels>=1) + eps)
    #print("Media")
    #print(medias[:,0])
    #precision should only count frames with foreground.
    media[3] = np.sum(medias[:,3], 0)/(np.sum(foreground_pixels>=1) + eps)
    #recall should only count frames with foreground.
    media[4] = np.sum(medias[:,4], 0)/(np.sum(foreground_pixels>=1) + eps)
    #accuracy should only count frames with foreground.
    #media(5) = np.sum(medias[:,5], 0)/(np.sum(foreground_pixels>=1) + eps)
    #fmeasure should only count frames with foreground.
    media[6] = np.sum(medias[:,6], 0)/(np.sum(foreground_pixels>=1) + eps)
    desv = np.std(medias[:,0:8], 0)
    desv[0] = np.std(medias[:,0][foreground_pixels>=1])
    desv[3] = np.std(medias[:,3][foreground_pixels>=1])
    desv[4] = np.std(medias[:,4][foreground_pixels>=1])
    desv[6] = np.std(medias[:,6][foreground_pixels>=1])
    
    return [medias,media,desv,numFrames,cont]
    

def evaluate_ours(evaluation_input_folder, evaluation_output_folder, encoder_model_version = 'models22', version = "version1", gaussian_noise_added_to_training_mu = 0, gaussian_noise_added_to_training_sigma = 0, image_gaussian_noise_mu = 0, image_gaussian_noise_sigma = 0, dataset_name_list = None, K_list = None, C_list = None, ALPHA_list = None, measures_list = None, segmentations_in_subfolders = False):

    if (dataset_name_list is None):
        dataset_name_list = ["canoe", "pedestrians", "port_0_17fps", "water_surface", "streetCornerAtNight", "tramStation","port_0_17fps", "overpass", "boats"]
        #dataset_name_list = ["canoe", "overpass"]
        dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
        #dataset_name_list = ["overpass","canoe", "port_0_17fps", "boats"]
        
    if (K_list is None):
        K_list = [2, 3, 4, 5, 6, 7, 8]

    if (C_list is None):
        C_list = [3, 6, 9, 12, 15]

    if (ALPHA_list is None):
        ALPHA_list = [0.001, 0.005, 0.01, 0.05]
        
    if (measures_list is None):
        measures_list = ['S', "precision", "recall", "accuracy", "fmeasure", "specificity"]

    total_segmentations_to_evaluate = len(dataset_name_list) * len(K_list) * len(C_list) * len(ALPHA_list)
    
    results = np.zeros ([len(dataset_name_list), len(K_list), len(C_list), len(ALPHA_list), 16])
    results -= 1
    
    evaluations_performed = 0
    index_dataset = 0
    for dataset_name in dataset_name_list:
        known_dataset = True
        count = 0
        if (dataset_name == 'water_surface'):

            ground_truth_dir = '../../data/water_surface/ground_truth'
            complente_GT_dataset = False
            prefix = 'segmented_img_f'
            sufix = '.bmp'
            
        elif (dataset_name == 'boats'):

            ground_truth_dir = '../../data/dynamicBackground/boats/groundtruth'
            dataset_subfolder = 'dynamicBackground'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == 'canoe'):

            ground_truth_dir = '../../data/dynamicBackground/canoe/groundtruth'
            dataset_subfolder = 'dynamicBackground'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == 'pedestrians'):

            ground_truth_dir = '../../data/baseline/pedestrians/groundtruth'
            dataset_subfolder = 'baseline'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == 'port_0_17fps'):

            ground_truth_dir = '../../data/lowFramerate/port_0_17fps/groundtruth'
            dataset_subfolder = 'lowFramerate'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == 'overpass'):

            ground_truth_dir = '../../data/dynamicBackground/overpass/groundtruth'
            dataset_subfolder = 'dynamicBackground'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'
            
        elif (dataset_name == "streetCornerAtNight"):
            
            ground_truth_dir = '../../data/nightVideos/streetCornerAtNight/groundtruth'
            dataset_subfolder = 'nightVideos'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == "tramStation"):

            ground_truth_dir = '../../data/nightVideos/tramStation/groundtruth'
            dataset_subfolder = 'nightVideos'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == "blizzard"):
        
            ground_truth_dir = '../../data/badWeather/blizzard/groundtruth'
            dataset_subfolder = 'badWeather'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'

        elif (dataset_name == "wetSnow"):
            
            ground_truth_dir = '../../data/badWeather/wetSnow/groundtruth'
            dataset_subfolder = 'badWeather'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'
            
        elif (dataset_name == "fountain01"):
            
            ground_truth_dir = '../../data/dynamicBackground/fountain01/groundtruth'
            dataset_subfolder = 'dynamicBackground'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'
            
        elif (dataset_name == "fountain02"):
            
            ground_truth_dir = '../../data/dynamicBackground/fountain02/groundtruth'
            dataset_subfolder = 'dynamicBackground'
            complente_GT_dataset = True
            prefix = 'segmented_img_in'
            sufix = '.jpg'
            
        else:

            print ('Unknown dataset', dataset_name)
            known_dataset = False
            
        print(ground_truth_dir)
        
        if (known_dataset):
            evaluation_dataset_output_folder = os.path.join(evaluation_output_folder, dataset_name)
            #if not os.path.exists(evaluation_dataset_output_folder):
                #os.makedirs (evaluation_dataset_output_folder)
                
            index_ALPHA = 0
            for ALPHA in ALPHA_list:
                index_K = 0    
                for K in K_list:
                    index_C = 0
                    for C in C_list:
                        
                        if (version != "version1"):
                            if (segmentations_in_subfolders):
                                segmented_images_dir = evaluation_input_folder + "/" + dataset_subfolder + "/" + dataset_name + '/segmentation_' + version + '_' + encoder_model_version
                            else:
                                segmented_images_dir = evaluation_input_folder + dataset_name + '/segmentation_' + version + '_' + encoder_model_version
                        else:
                            segmented_images_dir = evaluation_input_folder + dataset_name + '/segmentation_8x8_' + encoder_model_version
                           
                        if not (K is None):
                            segmented_images_dir = segmented_images_dir + '_K=' + str(K)
                        if not (C is None):
                            segmented_images_dir = segmented_images_dir + '_C=' + str(C)
                        if not (ALPHA is None):
                            segmented_images_dir = segmented_images_dir + '_ALPHA=' + str(ALPHA)
                        if not (gaussian_noise_added_to_training_mu is None):
                            segmented_images_dir = segmented_images_dir + '_T_MU=' + str(gaussian_noise_added_to_training_mu)
                        if not (gaussian_noise_added_to_training_sigma is None):
                            segmented_images_dir = segmented_images_dir + '_T_SIGMA=' + str(gaussian_noise_added_to_training_sigma)
                        if not (image_gaussian_noise_mu is None):
                            segmented_images_dir = segmented_images_dir + '_MU=' + str(image_gaussian_noise_mu)
                        if not (image_gaussian_noise_sigma is None):
                            segmented_images_dir = segmented_images_dir + '_SIGMA=' + str(image_gaussian_noise_sigma)
                        segmented_images_dir = segmented_images_dir + '/BW'                        
                        print(segmented_images_dir)
                        if (os.path.isdir(ground_truth_dir) and os.path.isdir(segmented_images_dir)):
                            count += 1
                            [medias,m,d,n,c] = evaluate_dataset(ground_truth_dir, segmented_images_dir, prefix = prefix, sufix = sufix, add_zeros_before_number= True)
                            results[index_dataset, index_K, index_C, index_ALPHA, 0:8] = m
                            results[index_dataset, index_K, index_C, index_ALPHA, 8:16] = d
                            
                            evaluations_performed +=1
                            print("OK - " + str(evaluations_performed) + "/" + str(total_segmentations_to_evaluate))
                             
                        index_C += 1
                        
                    index_K += 1
                    
                for measure in measures_list:
                    evaluation_image_dir = os.path.join(evaluation_dataset_output_folder, measure)
                    #if not(os.path.isdir (evaluation_image_dir)):
                        #os.makedirs (evaluation_image_dir)
                                       
                    evaluation_image = evaluation_image_dir + '/' + 'ALPHA=' + str(ALPHA) + '.png'
                    if (measure == "S"):
                        measure_index = 0
                        
                    elif (measure == "precision"):
                        measure_index = 3
                        
                    elif (measure == "recall"):
                        measure_index = 4
                        
                    elif (measure == "accuracy"):
                        measure_index = 5
                        
                    elif (measure == "fmeasure"):
                        measure_index = 6    
                        
                    elif (measure == "specificity"):
                        measure_index = 7
                        
                    #mat = np.reshape(results[index_dataset, :, :, index_ALPHA, measure_index], [len(K_list), len(C_list)])
                    
                    #surf(C_list, K_list, mat, 'FaceAlpha', 0.25)
                    #zlabel(measure)
                    #xlabel("C")
                    #ylabel("K")
                    #saveas(gcf, evaluation_image1)
                    
                    #mat = mat';
                    #clf
                    # [char(dataset_name) char(ALPHA)]
                    #surf(K_list, C_list, mat, 'FaceAlpha', 0.25)
                    #zlabel(measure)
                    #xlabel("K")
                    #ylabel("C")
                    #saveas(gcf, evaluation_image2)
                index_ALPHA += 1
            index_dataset += 1
    return results
        
def measureCDnet(result_path, groundtruth_path):

    eps = sys.float_info.epsilon
    
    BW = cv2.imread(result_path)
    GT = cv2.imread(groundtruth_path)
    GT = GT[:,:,0]
    BW = BW[:,:,0]

    BW = 255*(BW > 128)

    imGT = GT
    imBinary = BW

    TP = np.sum((imGT==255)&(imBinary==255))  # True Positive
    #TN = np.sum((imGT<=50)&(imBinary==0))    # True Negative
    #FP = np.sum((imGT<=50)&(imBinary==255))  # False Positive
    FN = np.sum((imGT==255)&(imBinary==0))    # False Negative

    OriginalFN = FN

    TN = np.sum((imGT==0)&(imBinary==0)) + np.sum((imGT==50)&(imBinary==0)) + np.sum((imGT==170)&(imBinary==0))           # True Negative
    FP = np.sum((imGT==0)&(imBinary==255)) + np.sum((imGT==50)&(imBinary==255)) + np.sum((imGT==170)&(imBinary==255))     # False Positive

    recall = float(TP) / (TP + FN + eps)
    specificity = float(TN) / (TN + FP + eps)
    FPR = float(FP) / (FP + TN + eps)
    FNR = float(FN) / (TP + FN + eps)
    PWC = 100.0 * (FN + FP) / (TP + FP + FN + TN + eps)
    precision = float(TP) / (TP + FP + eps)
    FMeasure = 2.0 * (recall * precision) / (recall + precision + eps)

    S = float(TP) / (TP + FN + FP + eps)
    accuracy = float(TP + TN) / (TP + FP + FN + TN + eps)
    fmeasure = FMeasure
    
    return [S, FP, FN, precision, recall, accuracy, fmeasure, specificity, TP, OriginalFN]

    #################################################################
    
def plot_various_bars(means, stds, colors = None, bar_width = 0.1, ylabel = "", title = "", xlabels = [], labels = [], autolabel = False, ylim = [0,1], dir_to_save = None):

    """
    ========
    Barchart
    ========

    A bar plot with errorbars and height labels on individual bars
    """
    import matplotlib.pyplot as plt

    N = len(means[0])
    M = len(means)
    ind = np.arange(N)  # the x locations for the groups
    
    if (colors is None):
        colors_aux = ['r', 'b', 'g', 'k', 'y', 'c', 'm']
        colors = colors_aux[:M]
        while (M > len(colors)):
            colors.extend(colors_aux[:(M-len(colors))])

    fig, ax = plt.subplots()
    
    rects = []
    
    for i in range(M):
        
        rects.append(ax.bar(ind + i*bar_width + 0.5*bar_width - (M-1)*bar_width/2, means[i], bar_width, color=colors[i], yerr=None))
        
    # add some text for labels, title and axes ticks
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + bar_width / 2)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(ylim)
    
    ax.legend(labels, loc='upper center', bbox_to_anchor=(0.5,-0.05), ncol = N)

    if (autolabel):
        for i in range(M):
            autolabel(rects[i])

    if (dir_to_save is None):
        plt.show()
    else:
        plt.savefig(dir_to_save)
    
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def evaluate_all_ours_version1_1(noise = "0", ALPHA_list = [0.001, 0.005, 0.01, 0.05], dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]):
    results = 4*[[]]
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    dataset_index = 0
    result_path = "../../segmentationResult/version1.1/evaluationResult/test/noise/"+noise+"/"
    for dataset_name in dataset_name_list:
        r = evaluate_ours("../../segmentationResult/version1.1/ours/"+noise+"/",
             result_path,
             encoder_model_version = 'models22', version = "version1.1", 
             dataset_name_list = [dataset_name], K_list = [None], C_list = [None], 
             ALPHA_list = ALPHA_list, image_gaussian_noise_mu = None,
             image_gaussian_noise_sigma = None, gaussian_noise_added_to_training_sigma = 0.2)
        row1.append(max(r[0, 0, 0, :, 0]))
        row2.append(max(r[0, 0, 0, :, 3]))
        row3.append(max(r[0, 0, 0, :, 4]))
        row4.append(max(r[0, 0, 0, :, 6]))
        dataset_index += 1
    
    results=[row1,row2,row3,row4]
    print(results)
    plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S", "precision", "recall", "fmeasure"], autolabel = False, dir_to_save=os.path.join(result_path,"evaluation64_0_2"))
    
def evaluate_all_ours_version2_1(noise = "0", ALPHA_list = [0.001, 0.005, 0.01, 0.05], dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]):
    results = 4*[[]]
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    dataset_index = 0
    result_path = "../../segmentationResult/version2.1/evaluationResult/test/noise/"+noise+"/"
    for dataset_name in dataset_name_list:
        r = evaluate_ours("../../segmentationResult/version2.1/ours/"+noise+"/",
             result_path,
             encoder_model_version = 'models24', version = "version2.1", 
             dataset_name_list = [dataset_name], K_list = [None], C_list = [None], 
             ALPHA_list = ALPHA_list, image_gaussian_noise_mu = None,
             image_gaussian_noise_sigma = None, gaussian_noise_added_to_training_sigma = 0.2)
        row1.append(max(r[0, 0, 0, :, 0]))
        row2.append(max(r[0, 0, 0, :, 3]))
        row3.append(max(r[0, 0, 0, :, 4]))
        row4.append(max(r[0, 0, 0, :, 6]))
        dataset_index += 1
    
    results=[row1,row2,row3,row4]
    print(results)
    plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S", "precision", "recall", "fmeasure"], autolabel = False, dir_to_save=os.path.join(result_path,"evaluation32_0_2"))
    
def evaluate_all_ours_version7_1(noise = "0", ALPHA_list = [0.001, 0.005, 0.01, 0.05], dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]):
    results = 4*[[]]
    row1 = []
    row2 = []
    row3 = []
    row4 = []
    dataset_index = 0
    result_path = "../../segmentationResult/version7.1/evaluationResult/test/noise/"+noise+"/"
    for dataset_name in dataset_name_list:
        r = evaluate_ours("../../segmentationResult/version7.1/ours/"+noise+"/",
             result_path,
             encoder_model_version = 'models27', version = "version7.1", 
             dataset_name_list = [dataset_name], K_list = [None], C_list = [None], 
             ALPHA_list = ALPHA_list, image_gaussian_noise_mu = None,
             image_gaussian_noise_sigma = None, gaussian_noise_added_to_training_sigma = 0.2)
        row1.append(max(r[0, 0, 0, :, 0]))
        row2.append(max(r[0, 0, 0, :, 3]))
        row3.append(max(r[0, 0, 0, :, 4]))
        row4.append(max(r[0, 0, 0, :, 6]))
        dataset_index += 1
    
    results=[row1,row2,row3,row4]
    print(results)
    plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S", "precision", "recall", "fmeasure"], autolabel = False, dir_to_save=os.path.join(result_path,"evaluation16_0_2"))
    
    
def evaluate_all_ours_version1(noise = "0", ALPHA_list = [0.001, 0.005, 0.01, 0.05], dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]):
    
    results = []
    dataset_index = 0
    result_path = "../../segmentationResult/version1/evaluationResult/test/noise/"+noise+"/"
    for dataset_name in dataset_name_list:
        r = evaluate_ours("../../segmentationResult/version1/ours/"+noise+"/",
             result_path,
             encoder_model_version = 'models22', version = "8x8",
             gaussian_noise_added_to_training_sigma = 0.05, 
             dataset_name_list = [dataset_name], 
             ALPHA_list = ALPHA_list)
             
        results.append(np.max(r[0, :, :, :, 0]))
        dataset_index += 1
    
    print(results)
    plt = plot_various_bars([results], [[0,0,0,0,0,0,0]], ['r'], bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S"], autolabel = False)
    plt.savefig(os.path.join(result_path),"S")
    
def get_evaluation(noise = "0", version = "version1.1", figure_name = None, segmented_images_subdir = "ours" , ALPHA_list = [0.001, 0.005, 0.01, 0.05], dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"], K_list = [None], C_list = [None], image_gaussian_noise_mu = None, image_gaussian_noise_sigma = None, gaussian_noise_added_to_training_sigma = None, gaussian_noise_added_to_training_mu = None, encoder_model_version = "models22"):
    
    results = 2*[[]]
    row1 = []
    row2 = []
    print ("Starting evaluation...")
    dataset_index = 0
    input_path = "../segmentationResult/"+ version +"/" + segmented_images_subdir + "/"+noise+"/"
    result_path = "../segmentationResult/" + version + "/evaluationResult/"+noise+"/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    for dataset_name in dataset_name_list:
        r = evaluate_ours(input_path, result_path,
             encoder_model_version = encoder_model_version, version = version, 
             dataset_name_list = [dataset_name], K_list = K_list, C_list = C_list, 
             ALPHA_list = ALPHA_list, image_gaussian_noise_mu = image_gaussian_noise_mu,
             image_gaussian_noise_sigma = image_gaussian_noise_sigma, gaussian_noise_added_to_training_sigma = gaussian_noise_added_to_training_sigma,
             gaussian_noise_added_to_training_mu=gaussian_noise_added_to_training_mu,
             segmentations_in_subfolders = True)
        row1.append(np.max(r[0, :, :, :, 0]))
        row2.append(np.max(r[0, :, :, :, 6]))
        dataset_index += 1
    
    results = [row1,row2]
    #print(results)
    if not (figure_name is None):
        plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S","F-measure"], autolabel = False, dir_to_save=os.path.join(result_path,figure_name))
    results = np.array(results)
    np.save(os.path.join(result_path, "results"), results)
    return results
    
def get_evaluation_all_ours():
    
    segmentations_in_subfolders = True
    
    noise_list = ["0", "0.1", "0.2", "0.31622776601683794", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression10", "compression5", "compression1"]
    dataset_name_list = ["overpass", "pedestrians", "canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
    ALPHA_list = [0.001, 0.005, 0.01, 0.05]
    version_list = ["version1.3", "version2.3", "version6.3", "version7.3"]
    
    evaluations_performed = 0
    total_segmentations_to_evaluate = len(version_list) * len(dataset_name_list) * len(noise_list) * len(ALPHA_list)  
    results = np.zeros ([len(version_list), len(dataset_name_list), len(noise_list), len(ALPHA_list), 16])
    results -= 1
    
    errors = []
    version_index = 0
    
    result_path = "../segmentationResult/ours/evaluationResult/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    for version in version_list:
    
        if (version == "version1.3"):
            encoder_model_version = "models22"
        if (version == "version2.3"):
            encoder_model_version = "models24"
        if (version == "version6.3"):
            encoder_model_version = "models26"
        if (version == "version7.3"):
            encoder_model_version = "models27"
            
        noise_index = 0
        
        for noise in noise_list:
        
            print ("Starting evaluation...")

            input_path = "../segmentationResult/"+ version +"/ours/" + noise + "/"

            dataset_index = 0
            for dataset_name in dataset_name_list:
                known_dataset = True
                if (dataset_name == 'water_surface'):

                    ground_truth_dir = '../../data/water_surface/ground_truth'
                    complente_GT_dataset = False
                    prefix = 'segmented_img_f'
                    sufix = '.bmp'
                        
                elif (dataset_name == 'boats'):

                    ground_truth_dir = '../../data/dynamicBackground/boats/groundtruth'
                    dataset_subfolder = 'dynamicBackground'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == 'canoe'):

                    ground_truth_dir = '../../data/dynamicBackground/canoe/groundtruth'
                    dataset_subfolder = 'dynamicBackground'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == 'pedestrians'):

                    ground_truth_dir = '../../data/baseline/pedestrians/groundtruth'
                    dataset_subfolder = 'baseline'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == 'port_0_17fps'):

                    ground_truth_dir = '../../data/lowFramerate/port_0_17fps/groundtruth'
                    dataset_subfolder = 'lowFramerate'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == 'overpass'):

                    ground_truth_dir = '../../data/dynamicBackground/overpass/groundtruth'
                    dataset_subfolder = 'dynamicBackground'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'
                        
                elif (dataset_name == "streetCornerAtNight"):
                        
                    ground_truth_dir = '../../data/nightVideos/streetCornerAtNight/groundtruth'
                    dataset_subfolder = 'nightVideos'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == "tramStation"):

                    ground_truth_dir = '../../data/nightVideos/tramStation/groundtruth'
                    dataset_subfolder = 'nightVideos'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == "blizzard"):
                   
                    ground_truth_dir = '../../data/badWeather/blizzard/groundtruth'
                    dataset_subfolder = 'badWeather'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'

                elif (dataset_name == "wetSnow"):
                        
                    ground_truth_dir = '../../data/badWeather/wetSnow/groundtruth'
                    dataset_subfolder = 'badWeather'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'
                        
                elif (dataset_name == "fountain01"):
                        
                    ground_truth_dir = '../../data/dynamicBackground/fountain01/groundtruth'
                    dataset_subfolder = 'dynamicBackground'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'
                        
                elif (dataset_name == "fountain02"):
                        
                    ground_truth_dir = '../../data/dynamicBackground/fountain02/groundtruth'
                    dataset_subfolder = 'dynamicBackground'
                    complente_GT_dataset = True
                    prefix = 'segmented_img_in'
                    sufix = '.jpg'
                        
                else:

                    print ('Unknown dataset', dataset_name)
                    known_dataset = False
                    
                if (noise == "compression"):
                    sufix = '.jpeg'
                        
                print(ground_truth_dir)
                    
                if (known_dataset):
                            
                    ALPHA_index = 0
                    for ALPHA in ALPHA_list:
                        
                        if (version != "version1"):
                            if (segmentations_in_subfolders):
                                segmented_images_dir = input_path + dataset_subfolder + "/" + dataset_name + '/segmentation_' + version + '_' + encoder_model_version
                            else:
                                segmented_images_dir = evaluation_input_folder + dataset_name + '/segmentation_' + version + '_' + encoder_model_version
                        else:
                            segmented_images_dir = evaluation_input_folder + dataset_name + '/segmentation_8x8_' + encoder_model_version
                                       
                        if not (ALPHA is None):
                            segmented_images_dir = segmented_images_dir + '_ALPHA=' + str(ALPHA)

                        segmented_images_dir = segmented_images_dir + '/BW'                        

                        print(segmented_images_dir)
                                
                        if (os.path.isdir(ground_truth_dir) and os.path.isdir(segmented_images_dir)):
                            [medias,m,d,n,c] = evaluate_dataset(ground_truth_dir, segmented_images_dir, prefix = prefix, sufix = sufix, add_zeros_before_number= True)
                            results[version_index, dataset_index, noise_index, ALPHA_index, 0:8] = m
                            results[version_index, dataset_index, noise_index, ALPHA_index, 8:16] = d
                                        
                            evaluations_performed +=1
                            print("OK - " + str(evaluations_performed) + "/" + str(total_segmentations_to_evaluate))
                            
                        else:
                            print("ERROR - " + str(evaluations_performed) + "/" + str(total_segmentations_to_evaluate))
                            errors.append([version, noise, dataset_name, ALPHA])

                        ALPHA_index += 1
                    
                dataset_index += 1
          
            noise_index += 1
        
        version_index += 1
            
                #r = evaluate_ours(input_path, result_path,
                #     encoder_model_version = encoder_model_version, version = version, 
                #     dataset_name_list = [dataset_name], K_list = K_list, C_list = C_list, 
                #     ALPHA_list = ALPHA_list, image_gaussian_noise_mu = image_gaussian_noise_mu,
                #     image_gaussian_noise_sigma = image_gaussian_noise_sigma, gaussian_noise_added_to_training_sigma = gaussian_noise_added_to_training_sigma,
                #     gaussian_noise_added_to_training_mu=gaussian_noise_added_to_training_mu,
                #     segmentations_in_subfolders = True)
                #row1.append(np.max(r[0, :, :, :, 0]))
                #row2.append(np.max(r[0, :, :, :, 6]))
                #dataset_index += 1
            
            #results = [row1,row2]
            #print(results)
            #if not (figure_name is None):
                #plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S","F-measure"], autolabel = False, dir_to_save=os.path.join(result_path,figure_name))
            #results = np.array(results)
    np.save(os.path.join(result_path, "results"), results)
    return [results, errors]
            #get_evaluation(noise = noise, version = version, dataset_name_list = dataset_name_list, encoder_model_version=encoder_model_version, ALPHA_list = ALPHA_list)
            
def get_evaluation_rivals(noise_level_list = ["0", "0.1", "0.2", "0.31", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform", "compression10", "compression5", "compression1"], compare_methods = ["zivkovic", "kde", "wren", "sobs", "FSOM"], figure_name = None, segmented_images_subdir = "ours", dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]):

    count = 0
    method_index = 0
    result_path = "../segmentationResult/rivals/evaluationResult/"
    
    results = np.zeros([len(compare_methods), len(dataset_name_list), len(noise_level_list), 16])
    results -= 1
    
    errors = []
    
    total_number_of_evaluations = len(compare_methods) * len(dataset_name_list) * len(noise_level_list)
    
    for method_name in compare_methods:           
            
        if (method_name == "wren"):
            prefix = "WrenGA_Unif_0.00_"
            suffix = ".png"
            addZerosBeforeNumber = False
                
        elif (method_name == "zivkovic"):
            prefix = "ZivkovicGMM_Unif_0.00_"
            suffix = ".png"
            addZerosBeforeNumber = False
                
        elif (method_name == "sobs"):
            prefix = "AdaptiveSOM_Unif_0.00_"
            suffix = ".png"
            addZerosBeforeNumber = False
                
        elif (method_name == "kde"):
            prefix = "KDE_Unif_0.00_"
            suffix = ".png"
            addZerosBeforeNumber = False
                
        elif (method_name == "FSOM"):
            prefix = "seg"
            suffix = ".jpg"
            addZerosBeforeNumber = True
        
        dataset_index = 0
        for dataset_name in dataset_name_list:
            known_dataset = True
            
            if (dataset_name == 'water_surface'):

                ground_truth_dir = '../../data/water_surface/ground_truth'
                
            elif (dataset_name == 'boats'):

                ground_truth_dir = '../../data/dynamicBackground/boats/groundtruth'
                dataset_subfolder = 'dynamicBackground'

            elif (dataset_name == 'canoe'):

                ground_truth_dir = '../../data/dynamicBackground/canoe/groundtruth'
                dataset_subfolder = 'dynamicBackground'

            elif (dataset_name == 'pedestrians'):

                ground_truth_dir = '../../data/baseline/pedestrians/groundtruth'
                dataset_subfolder = 'baseline'

            elif (dataset_name == 'port_0_17fps'):

                ground_truth_dir = '../../data/lowFramerate/port_0_17fps/groundtruth'
                dataset_subfolder = 'lowFramerate'

            elif (dataset_name == 'overpass'):

                ground_truth_dir = '../../data/dynamicBackground/overpass/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            elif (dataset_name == "streetCornerAtNight"):
                
                ground_truth_dir = '../../data/nightVideos/streetCornerAtNight/groundtruth'
                dataset_subfolder = 'nightVideos'

            elif (dataset_name == "tramStation"):

                ground_truth_dir = '../../data/nightVideos/tramStation/groundtruth'
                dataset_subfolder = 'nightVideos'

            elif (dataset_name == "blizzard"):
            
                ground_truth_dir = '../../data/badWeather/blizzard/groundtruth'
                dataset_subfolder = 'badWeather'

            elif (dataset_name == "wetSnow"):
                
                ground_truth_dir = '../../data/badWeather/wetSnow/groundtruth'
                dataset_subfolder = 'badWeather'
                
            elif (dataset_name == "fountain01"):
                
                ground_truth_dir = '../../data/dynamicBackground/fountain01/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            elif (dataset_name == "fountain02"):
                
                ground_truth_dir = '../../data/dynamicBackground/fountain02/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            else:
            
                known_Dataset = False
            
            noise_index = 0
            
            print(ground_truth_dir)
            
            for noise in noise_level_list:
                
                if (noise == "0" and method_name == "FSOM"): #We need to establish this exception.
                    prefix = "bin"
                    suffix = ".png"
                if (noise != "0" and method_name == "FSOM"): #We need to establish this exception.
                    prefix = "seg"
                    suffix = ".jpg"
                    
                input_path = "../segmentationResult/caepia18_m/" + method_name + "/" + method_name + "_" + noise + "/" + dataset_subfolder + "/" + dataset_name
                
                print(input_path)
                if os.path.exists(input_path):

                    [medias,m,d,n,c] = evaluate_dataset(ground_truth_dir, input_path, prefix = prefix, sufix = suffix, add_zeros_before_number = addZerosBeforeNumber)
                    results[method_index, dataset_index, noise_index, 0:8] = m
                    results[method_index, dataset_index, noise_index, 8:16] = d
                    count +=1
                    
                    print("OK " + str(count) + "/" + str(total_number_of_evaluations))
                else:
                    print("DIRECTORY ERROR " + str(count) + "/" + str(total_number_of_evaluations))
                    errors.append([method_name, dataset_name, noise])
                
                noise_index += 1
                    
            dataset_index += 1
            
        method_index += 1   
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #print(results)
    if not (figure_name is None):
        plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S","F-measure"], autolabel = False, dir_to_save=os.path.join(result_path,figure_name))
    results
    np.save(os.path.join(result_path, "results"), results)
    return [results, errors]
    
def get_evaluation_SOBS(noise_level_list = ["mask", "mask2x2", "mask3x3", "mask4x4"], compare_methods = ["sobs"], figure_name = None, segmented_images_subdir = "ours", dataset_name_list = ["pedestrians"]):

    count = 0
    method_index = 0
    result_path = "../segmentationResult/rivals/sobsTest/"
    
    results = np.zeros([len(compare_methods), len(dataset_name_list), len(noise_level_list), 16])
    results -= 1
    
    errors = []
    
    total_number_of_evaluations = len(compare_methods) * len(dataset_name_list) * len(noise_level_list)
    
    for method_name in compare_methods:           

        if (method_name == "sobs"):
            prefix = "bin"
            suffix = ".png"
            addZerosBeforeNumber = True

        
        dataset_index = 0
        for dataset_name in dataset_name_list:
            known_dataset = True
            
            if (dataset_name == 'water_surface'):

                ground_truth_dir = '../../data/water_surface/ground_truth'
                
            elif (dataset_name == 'boats'):

                ground_truth_dir = '../../data/dynamicBackground/boats/groundtruth'
                dataset_subfolder = 'dynamicBackground'

            elif (dataset_name == 'canoe'):

                ground_truth_dir = '../../data/dynamicBackground/canoe/groundtruth'
                dataset_subfolder = 'dynamicBackground'

            elif (dataset_name == 'pedestrians'):

                ground_truth_dir = '../../data/baseline/pedestrians/groundtruth'
                dataset_subfolder = 'baseline'

            elif (dataset_name == 'port_0_17fps'):

                ground_truth_dir = '../../data/lowFramerate/port_0_17fps/groundtruth'
                dataset_subfolder = 'lowFramerate'

            elif (dataset_name == 'overpass'):

                ground_truth_dir = '../../data/dynamicBackground/overpass/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            elif (dataset_name == "streetCornerAtNight"):
                
                ground_truth_dir = '../../data/nightVideos/streetCornerAtNight/groundtruth'
                dataset_subfolder = 'nightVideos'

            elif (dataset_name == "tramStation"):

                ground_truth_dir = '../../data/nightVideos/tramStation/groundtruth'
                dataset_subfolder = 'nightVideos'

            elif (dataset_name == "blizzard"):
            
                ground_truth_dir = '../../data/badWeather/blizzard/groundtruth'
                dataset_subfolder = 'badWeather'

            elif (dataset_name == "wetSnow"):
                
                ground_truth_dir = '../../data/badWeather/wetSnow/groundtruth'
                dataset_subfolder = 'badWeather'
                
            elif (dataset_name == "fountain01"):
                
                ground_truth_dir = '../../data/dynamicBackground/fountain01/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            elif (dataset_name == "fountain02"):
                
                ground_truth_dir = '../../data/dynamicBackground/fountain02/groundtruth'
                dataset_subfolder = 'dynamicBackground'
                
            else:
            
                known_Dataset = False
            
            noise_index = 0
            
            print(ground_truth_dir)
            
            for noise in noise_level_list:
                
                if (noise == "0" and method_name == "FSOM"): #We need to establish this exception.
                    prefix = "bin"
                    suffix = ".png"
                if (noise != "0" and method_name == "FSOM"): #We need to establish this exception.
                    prefix = "seg"
                    suffix = ".jpg"
                    
                input_path = "../segmentationResult/testSOBS/" + noise
                
                print(input_path)
                if os.path.exists(input_path):

                    [medias,m,d,n,c] = evaluate_dataset(ground_truth_dir, input_path, prefix = prefix, sufix = suffix, add_zeros_before_number = addZerosBeforeNumber)
                    results[method_index, dataset_index, noise_index, 0:8] = m
                    results[method_index, dataset_index, noise_index, 8:16] = d
                    count +=1
                    
                    print("OK " + str(count) + "/" + str(total_number_of_evaluations))
                else:
                    print("DIRECTORY ERROR " + str(count) + "/" + str(total_number_of_evaluations))
                    errors.append([method_name, dataset_name, noise])
                
                noise_index += 1
                    
            dataset_index += 1
            
        method_index += 1   
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    #print(results)
    if not (figure_name is None):
        plot_various_bars(results, None, bar_width = 0.1, ylabel = "", title = "", xlabels = dataset_name_list, labels = ["S","F-measure"], autolabel = False, dir_to_save=os.path.join(result_path,figure_name))
    results
    np.save(os.path.join(result_path, "results"), results)
    return [results, errors]
    
def generate_evaluation_bars():
    
    rivals_results_path = "../segmentationResult/rivals/evaluationResult/results.npy"
    our_results_path = "../segmentationResult/ours/evaluationResult/results.npy"
    dataset_name_list = ["overpass","pedestrians","canoe", "port_0_17fps", "boats", "fountain01", "fountain02"]
    ALPHA_list = [0.001, 0.005, 0.01, 0.05]
    version_list = ["version1.2", "version2.2", "version6.2", "version7.2"]
    
    noise_level_list = ["0", "0.1", "0.2", "0.31", "mask", "mask2x2", "mask3x3", "mask4x4", "saltpepper", "uniform"]
    compare_methods = ["zivkovic", "kde", "wren", "sobs", "FSOM"]
    
    if (os.path.exists(rivals_results_path)):
        print("Loading rivals results")
        rivals_results = np.load(rivals_results_path)
    else:
        print("Evaluating rivals to get results.")
        rivals_results = get_evaluation_rivals()
    if (os.path.exists(our_results_path)):
        print("Loading our results")
        our_results = np.load(our_results_path)
    else:
        print("Evaluating ours to get results.")
        our_results = get_evaluation_all_ours()
    
    print(our_results.shape)
    print(rivals_results.shape)
    
    for noise_index in range(len(noise_level_list)):
        noise = noise_level_list[noise_index]
        noise = noise.replace(".", "_")
            
        for version_index in range(len(version_list)):
            #S_list = (1+len(compare_methods))*[[]]
            S_list = [[],[],[],[],[],[]]
            for dataset_name_index in range(len(dataset_name_list)):                    
                S_list_index = 0
                S_list[S_list_index].append(np.max(our_results[version_index, dataset_name_index, noise_index, :, 0]))
                S_list_index += 1
                
                for compare_method_index in range(len(compare_methods)):
                    
                    S_list[S_list_index].append(np.max(rivals_results[compare_method_index, dataset_name_index, noise_index, 0]))
                    S_list_index += 1        
        
            version_str = version_list[version_index].replace(".", "_")
            print(np.array(S_list).shape)
            plot_various_bars(S_list, None, ylabel = "S", title = version_str + "_" + noise, xlabels = dataset_name_list, labels = [version_str]+compare_methods, autolabel = False, ylim = [0,1], dir_to_save = os.path.join("../segmentationResult/ours/evaluationResult", version_str + "_" + noise))
            
            
def get_n_bests_from_dataset(path_GT, path_BW, number_of_images = 10, minimum_foreground_pixels = 50, prefix = None, sufix = None, add_zeros_before_number = False):

    n_max = []
    n_max_names = []

    if (path_GT[-1] != '/'):
        path_GT = path_GT + '/'
        
    if (path_BW[-1] != '/'):
        path_BW = path_BW + '/'
        
    info_files_GT = os.listdir(path_GT)
    info_files_GT = sorted(info_files_GT)
    info_files_BW = os.listdir(path_BW)
    info_files_BW = sorted(info_files_BW)
    medias = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    cont = 0
    numFrames = 0


    #We must get the temporal roi information if it exists.
    [gt_path_main_folder, _] = os.path.split(path_GT[:-1])
    temporal_roi_file = os.path.join(gt_path_main_folder, 'temporalROI.txt')
    if (os.path.isfile(temporal_roi_file)):
        #If it exists, we will use it as loop initiallization point.
        roi_info_file = open(temporal_roi_file, "r")
        roi_info = roi_info_file.readline()
        [first_element_in_roi, last_element_in_roi] = str.split(roi_info) #Info about the first and the last frame to study.
        first_element_in_roi = int(first_element_in_roi)
        last_element_in_roi = int(last_element_in_roi)
    else:
        first_element_in_roi = 1

    for i in range(len(info_files_GT)):
        if (os.path.isfile(os.path.join(path_GT, info_files_GT[i]))):
            #Is a file.
            filename_GT = info_files_GT[i]
            [_, name_ext] = os.path.split(filename_GT)
            [name, ext] = os.path.splitext(name_ext)
            image_number = int(name[2:])
            if (image_number >= first_element_in_roi):
                # We get those with .bmp, .jpg or .png extensions.
                if ((ext == '.bmp') | (ext == '.jpg') | (ext == '.png')):
                    path_img_GT = os.path.join(path_GT, filename_GT)
                    numFrames += 1
            
                    prefix_0s_length = len(name[2:]) - len(str(image_number))
                    zeros_string = ""
                    if (add_zeros_before_number):
                        for j in range(prefix_0s_length):
                            zeros_string = zeros_string + '0'
                    filename_bw = prefix + zeros_string + str(image_number) + sufix

                    path_img_BW = os.path.join(path_BW, filename_bw)
                    
                    if(os.path.isfile(path_img_BW)): #If its segmented image does exist
                        if(cont == 0):
                            medias = np.array([[-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.]])
                        else:
                            medias = np.vstack((medias, np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])))
                        print(path_img_BW)
                        print(path_img_GT)
                        medias[cont,0:10] = measureCDnet(path_img_BW, path_img_GT)
                        medias[cont,10] = i
                        if(medias[cont, 8] + medias[cont, 9] >= minimum_foreground_pixels):
                            if (len(n_max) < number_of_images):
                                n_max.append(medias[cont, 6])
                                n_max_names.append(filename_GT)
                            else:
                                current_min = min(n_max)
                                current_min_index = n_max.index(current_min)
                                current_min_name = n_max_names[current_min_index]
                                if(current_min < medias[cont,6]):
                                    n_max[current_min_index] = medias[cont,6]
                                    n_max_names[current_min_index] = filename_GT
                        cont = cont + 1

    
    return [n_max, n_max_names]
