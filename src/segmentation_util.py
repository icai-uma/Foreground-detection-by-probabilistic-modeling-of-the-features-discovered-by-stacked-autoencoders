import numpy as np
import cv2
import util
import math
import time

## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

def segmentation_dense_autoencoder_by_patches(result_path, version, N, dataset_dir, L = None, segmented_output = True, encoder_model_version = None, ALPHA = None, K = None, C = None, Alpha = None, gaussian_noise_added_to_training_mu = None, gaussian_noise_added_to_training_sigma = None, image_gaussian_noise_mu = None, image_gaussian_noise_sigma = None, output_video = False):

    if (encoder_model_version is None):
        if (version == "version1.2"):
            encoder_model_version = "models22"
        if (version == "version2.2"):
            encoder_model_version = "models24"
        if (version == "version6.2"):
            encoder_model_version = "models26"
        if (version == "version7.2"):
            encoder_model_version = "models27"

    #L is the narrowest encoder layer size.
    if (L is None):
        if (version == "version1.2"):
            L = 64
        if (version == "version2.2"):
            L = 32
        if (version == "version6.2"):
            L = 128
        if (version == "version7.2"):
            L = 16
    
    #We specify the result path.
    result_path = result_path + "_" + version + "_" + encoder_model_version

    if not (K is None):
        result_path = result_path + '_K=' + str(K)
    if not (C is None):
        result_path = result_path + '_C=' + str(C)
    if not (ALPHA is None):
        result_path = result_path + '_ALPHA=' + str(ALPHA)
    if not (gaussian_noise_added_to_training_mu is None):
        result_path = result_path + '_T_MU=' + str(gaussian_noise_added_to_training_mu)
    if not (gaussian_noise_added_to_training_sigma is None):
        result_path = result_path + '_T_SIGMA=' + str(gaussian_noise_added_to_training_sigma)
    if not (image_gaussian_noise_mu is None):
        result_path = result_path + '_MU=' + str(image_gaussian_noise_mu)
    if not (image_gaussian_noise_sigma is None):
        result_path = result_path + '_SIGMA=' + str(image_gaussian_noise_sigma)

    # #Do We generate a video?
    # output_video = False
    
    # #Do we generate BW images?
    # segmented_output = True

    patch_height = 16
    patch_width = 16
    slicing_window_height = 8
    slicing_window_width = 8

    if (segmented_output):
        result_segmented = os.path.join(result_path, "BW")

    if not os.path.exists(result_path): #If result path does not exist, we create it.
        os.makedirs( result_path )

    if (segmented_output):
        if not os.path.exists(result_segmented): #If result path does not exist, we create it.
            os.makedirs( result_segmented )

    print ("Loading training data.")
    dataset_dirs = os.listdir(dataset_dir)
    dataset_dirs = sorted(dataset_dirs)
    original_dataset_size = len(dataset_dirs)

    #We get an image from the dataset to obtain images height and width.
    if original_dataset_size > 0:
        img_name = dataset_dirs[0]
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread( img_dir )
        original_image_height = img.shape[0]
        original_image_width = img.shape[1]
        original_image_height_with_padding = original_image_height + 8
        original_image_width_with_padding = original_image_width + 8

    patches_per_image = int((original_image_height - patch_height) / slicing_window_height + 2) * int((original_image_width - patch_width) / slicing_window_width + 2)

    if (output_video):
        #Output Video initialization.
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        processed_video = cv2.VideoWriter(os.path.join(result_path, 'segmentated_video_' + str(dataset_name) + '.avi'),fourcc,int(fps),(original_image_width,original_image_height))
        
    if (not os.path.exists("./GENERAL_MU_"+str(L)+".npy")): #Load data from file to reduce time.
        print ("Extracting gaussian parameters from encoded images.")
        GENERAL_MU, GENERAL_SIGMA2 = util.extract_gaussian_parameters_from_encoded_images(encoder_model_path, encode_training_dataset, L)
        np.save("GENERAL_MU_"+str(L)+".npy", GENERAL_MU)
        np.save("GENERAL_SIGMA2_"+str(L)+".npy", GENERAL_SIGMA2)
    else:
        print ("Loading guassian parameters from encoded images.")
        GENERAL_MU = np.load("GENERAL_MU_"+str(L)+".npy")
        GENERAL_SIGMA2 = np.load("GENERAL_SIGMA2_"+str(L)+".npy")

    print("GENERAL_MU:")
    print(GENERAL_MU)

    print("GENERAL_SIGMA2:")
    print(GENERAL_SIGMA2)

    #There will be a gaussian distribution for each encoded patch component.
    SIGMA2 = np.zeros([int((original_image_height - patch_height) / slicing_window_height + 2), int((original_image_width - patch_width) / slicing_window_width + 2), 64])
    MU = np.zeros([int((original_image_height - patch_height) / slicing_window_height + 2), int((original_image_width - patch_width) / slicing_window_width + 2), 64])

    #Normalized training dataset matrix initialization.
    normalized_training_dataset = np.zeros([N, original_image_height_with_padding, original_image_width_with_padding, 3])

    index = 0
    for img_name in dataset_dirs[:N]:
        img_dir = os.path.join(dataset_dir, img_name)
        img = cv2.imread( img_dir )
        normalized_training_dataset[index, 4:original_image_height_with_padding-4, 4:original_image_width_with_padding-4] = img.astype( float ) / 255.
        index += 1

    #normalized_training_dataset = original_dataset / 255. #We normalize over [0,1] to work with the encoder model.
    #original_dataset = None #We delete the reference to array in memory to liberate memory.
    
    if (not (image_gaussian_noise_sigma is None)) and (not (image_gaussian_noise_mu is None)) and (image_gaussian_noise_sigma != 0 or image_gaussian_noise_mu != 0):
        print("Adding disruptive gaussian noise to training dataset with mu = " + str(image_gaussian_noise_mu) + " and sigma = " + str(image_gaussian_noise_sigma))
        normalized_training_dataset_with_noise = util.add_additive_gaussian_noise(normalized_training_dataset, image_gaussian_noise_mu, image_gaussian_noise_sigma, [0,8,8,0]) #Add gaussian noise to input images.
        normalized_training_dataset_with_noise = np.clip(normalized_training_dataset_with_noise, 0, 1, out=normalized_training_dataset_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1
    else:
        normalized_training_dataset_with_noise = normalized_training_dataset

    normalized_training_dataset = None #We delete the reference to array in memory to liberate memory.

    if (not (gaussian_noise_added_to_training_sigma is None)) and (not (gaussian_noise_added_to_training_mu is None)) and (gaussian_noise_added_to_training_sigma != 0 or gaussian_noise_added_to_training_mu != 0):
        print("Adding training gaussian noise with mu = " + str(gaussian_noise_added_to_training_mu) + " and sigma = " + str(gaussian_noise_added_to_training_sigma))
        training_data = util.add_additive_gaussian_noise(normalized_training_dataset_with_noise, gaussian_noise_added_to_training_mu, gaussian_noise_added_to_training_sigma, [0,8,8,0]) #Add gaussian noise to training images.
        training_data = np.clip(training_data, 0, 1, out=training_data) #We ensure that there is no value smaller than 0 and no value greater than 1
    else:
        training_data = normalized_training_dataset_with_noise[:N]

    normalized_training_dataset_with_noise = None #We delete the reference to array in memory to liberate memory.

    print ("Transforming training data to patches.")
    patched_dataset = util.split_in_patches_with_windows_various_images(training_data, patch_height, patch_width, slicing_window_height, slicing_window_width)
    patched_dataset_reshaped = np.reshape(patched_dataset, [patched_dataset.shape[0], patch_height*patch_width*3])
    patched_dataset = None #We delete the reference to array in memory to liberate memory.
    training_data = None #We delete the reference to array in memory to liberate memory.

    print ("Encoding initial " + str(N) + " images to obtain initial gaussian models.")
    predicted_patches = encoder_model.predict(patched_dataset_reshaped)
    patched_dataset_reshaped = None #We delete the reference to array in memory to liberate memory.

    print ("Obtaining initial median and standard desviation for each component.")

    for ii in range(MU.shape[0]):       #For each patch...
        for jj in range(MU.shape[1]):
            absolute_index = ii * MU.shape[1] + jj

            for kk in range(MU.shape[2]):   #For each component...
                MU[ii, jj, kk] = np.median(predicted_patches[absolute_index::patches_per_image, kk])
                SIGMA2[ii, jj, kk] = np.sum(np.power(predicted_patches[absolute_index::patches_per_image, kk] - MU[ii, jj, kk], 2)) / N

    print ("Starting incoming images processing.")

    #print ("Encoding final " + str(original_dataset_size - N) + " images to segment.")
    #predicted_patches = encoder_model.predict(patched_dataset_reshaped[N*patches_per_image:])

    predicted_patches = None #We delete the reference to array in memory to liberate memory.

    times_for_each_segmentation = np.zeros([original_dataset_size - N]) #An array to contain how much time we spend in each segmentation.
    print("Applying segmentation.")
    tt = 0
     
    for img_name in dataset_dirs[N:]: #for each new image...

        #We reaed the image.
        img_dir = os.path.join(dataset_dir, img_name)
        original_image = cv2.imread( img_dir ).astype( float )
        #We add pading and normalize it.
        normalized_image = np.zeros([1, original_image_height_with_padding, original_image_width_with_padding, 3])
        normalized_image[0, 4:original_image_height_with_padding-4, 4:original_image_width_with_padding-4] = original_image / 255.

        initial_time = time.time() #Start segmentation time.

        #If there is disruptive gaussian noise, we add it.
        if (image_gaussian_noise_mu > 0 or image_gaussian_noise_sigma > 0):
            normalized_image_with_noise = util.add_additive_gaussian_noise(normalized_image, image_gaussian_noise_mu, image_gaussian_noise_sigma, [0,8,8,0]) #Add gaussian noise to input images.
            normalized_image_with_noise = np.clip(normalized_image_with_noise, 0, 1, out=normalized_image_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1
            normalized_image_with_noise = np.reshape(normalized_image_with_noise, normalized_image_with_noise.shape[1:]) #We delete the first dimension that will always have length 1.
        else:
            normalized_image_with_noise = np.reshape(normalized_image, normalized_image.shape[1:]) #We delete the first dimension that will always have length 1.

        #We split image.
        patched_image = util.split_in_patches_with_windows_one_image(normalized_image_with_noise, patch_height, patch_width, slicing_window_height, slicing_window_width)
        
        initial_time = time.time() #Start segmentation time.

        #We reshape it.
        patched_image_reshape = np.reshape(patched_image, [patched_image.shape[0], patch_height*patch_width*3])
        #print(time.time()-initial_time)
        #We execute the prediction.
        image_patches_set = encoder_model.predict(patched_image_reshape)
        foreground_matrix_to_show = np.zeros([int(original_image_height/slicing_window_height), int(original_image_width/slicing_window_width)])
        #print(time.time()-initial_time)

        if (debug):
            aux_index_to_plot = 0

        #print("-")
        #print(time.time()-initial_time)

        for ii in range(MU.shape[0]):      #For each patch...
            for jj in range(MU.shape[1]):
                absolute_index = ii * MU.shape[1] + jj
                #We need to know if the patch is clasified as foreground or background.
                patch_value = image_patches_set[absolute_index]

                log_prob_v_cond_Fore = -L/2.*math.log(2*math.pi) - np.sum(np.log(np.sqrt(GENERAL_SIGMA2))) - 1/2.*np.sum(np.divide(np.power(patch_value - GENERAL_MU, 2), GENERAL_SIGMA2))
                log_prob_v_cond_Back = -L/2.*math.log(2*math.pi) - np.sum(np.log(np.sqrt(SIGMA2[ii,jj]))) - 1/2.*np.sum(np.divide(np.power(patch_value - MU[ii, jj], 2), SIGMA2[ii, jj]))

                if (log_prob_v_cond_Back >= log_prob_v_cond_Fore):
                    prob_Back_cond_v = 1./(1 + math.pow(math.e, log_prob_v_cond_Fore - log_prob_v_cond_Back))
                    prob_Fore_cond_v = (1 - prob_Back_cond_v)
                    
                if (log_prob_v_cond_Back < log_prob_v_cond_Fore):
                    prob_Fore_cond_v = 1./(1 + math.pow(math.e, log_prob_v_cond_Back - log_prob_v_cond_Fore))
                    prob_Back_cond_v = (1 - prob_Fore_cond_v)

                foreground_matrix_to_show[ii, jj] += prob_Fore_cond_v
                SIGMA2[ii,jj] = (1 - ALPHA * prob_Back_cond_v) * SIGMA2[ii,jj] + ALPHA * prob_Back_cond_v * np.power(patch_value - MU[ii,jj], 2)
                MU[ii,jj] = (1 - ALPHA * prob_Back_cond_v) * MU[ii,jj] + ALPHA * prob_Back_cond_v * patch_value

        #print(time.time()-initial_time)

        final_time = time.time() - initial_time
        times_for_each_segmentation[tt] = final_time
        output_image = util.draw_rectangles_over_image(normalized_image_with_noise[4:original_image_height+4, 4:original_image_width+4,:] * 255., foreground_matrix_to_show, 0.5, [0,0,0])
        #cv2.imwrite( os.path.join(result_path, 'segmented_img_' + img_name), output_image) #Create same image with foreground marked.
        if (segmented_output):
            #cv2.imwrite( os.path.join(result_segmented, 'segmented_img_' + img_name), util.create_segmented_image_from_foreground_matrix(foreground_matrix_to_show, 8, 8, 0.5, original_image_height, original_image_width))
            cv2.imwrite( os.path.join(result_segmented, 'segmented_img_' + img_name), util.create_segmented_grayscale_image_from_foreground_matrix(foreground_matrix_to_show, 8, 8, original_image_height, original_image_width))
        if (output_video):
            processed_video.write(output_image.astype('uint8'))
        tt += 1

    average_time = np.sum(times_for_each_segmentation)/times_for_each_segmentation.shape[0]
    print("End of segmentation. Average segmentation time: " + str(average_time))

    if (output_video):
        processed_video.release()
        
    #We will simply return result path as result.
    return result_path

