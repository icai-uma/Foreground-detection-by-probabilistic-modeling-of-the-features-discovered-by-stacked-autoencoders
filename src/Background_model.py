import numpy as np
import cv2
import util                                 # pylint: disable=import-error
import os
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm                       # pylint: disable=import-error


class Background_model:
    encoder = None                                          # encoder object.
    narrowest_layer_size = -1                               # narrowest layer size. It must bee qual to encoder output layer.
    padding_value = -1                                      # Value to serve as paddding to increase image size in order to have margin to get patches (example: we have a 16x16 patch but we segmented with 8x8 resolution, it is needed 8 as padding).
    SIGMA2 = None                                           # variance values for each component for each patch.
    MU = None                                               # median values for each component for each patch.
    FOREGROUND_MU = None                                    # median values for each component to be used as foreground to compare.
    FOREGROUND_SIGMA2 = None                                # variance values for each component to be used as foreground to compare.
    ALPHA = -1                                              # Background model update step size.
    normalized_training_dataset = None                      # matrix to contain data to be initialize background model.

    foreground_matrix_to_show = None                        # last segmented image.

    original_image_height = -1                              # Image height.
    original_image_width = -1                               # Image Width.
    patch_height = -1                                       # Patch height value.
    patch_width = -1                                        # Patch width value.
    patches_per_image = -1                                  # Number of patches per image.
    slicing_window_height = -1                              # Defines how patch moves through image.
    slicing_window_width = -1                               # Defines how patch moves through image.
    number_of_images_for_training = -1                      # Number of images that will be used to initialize the background model.

    training_done = False                                   # Control value to know if training is ended.
    data_initialized = False                                # Controlv alue to know if data has been initialized.

    processed_images = -1

    def __init__(self, encoder=None, narrowest_layer_size=-1, padding_value = -1, patch_height = -1, patch_width = -1, slicing_window_height = -1, slicing_window_width = -1, number_of_images_for_training = -1):            # Class to represent the background model base class for Foreground detection by probabilistic modeling of the features discovered by stackeddenoising autoencoders in noisy video sequences.
        self.encoder = encoder
        self.narrowest_layer_size = narrowest_layer_size
        self.padding_value = padding_value
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.slicing_window_height = slicing_window_height
        self.slicing_window_width = slicing_window_width
        self.training_done = False
        self.data_initialized = False
        self.number_of_images_for_training = number_of_images_for_training
        self.processed_images = 0
        self.patches_per_image = int((self.original_image_height - self.patch_height) / self.slicing_window_height + 2) * int((self.original_image_width - self.patch_width) / self.slicing_window_width + 2)
        self.foreground_matrix_to_show = None

    def initialize_data(self, image):                                                               # Method to initialize data.
        self.original_image_height = image.shape[0]                                                 # We set images height.  
        self.original_image_width = image.shape[1]                                                  # We set images width.
        original_image_height_with_padding = self.original_image_height + self.padding_value        # We obtain hegith with padding.
        original_image_width_with_padding = self.original_image_width + self.padding_value          # We obtain width with padding.
        #There will be a gaussian distribution for each encoded patch component.
        self.SIGMA2 = np.zeros(
            [int((self.original_image_height - self.patch_height) / self.slicing_window_height + 2),
             int((self.original_image_width - self.patch_width) / self.slicing_window_width + 2), 
             self.narrowest_layer_size])                                                            # We generate a matrix full of zeroes to initialize sigma2.
        self.MU = np.zeros(
            [int((self.original_image_height - self.patch_height) / self.slicing_window_height + 2),
             int((self.original_image_width - self.patch_width) / self.slicing_window_width + 2),
             self.narrowest_layer_size])                                                            # We generate a matrix full of zeroes to initialize mu.
        #Normalized training dataset matrix initialization.
        self.normalized_training_dataset = np.random.uniform(0, 1, 
            [self.number_of_images_for_training, original_image_height_with_padding, 
            original_image_width_with_padding, 3])                                                  # We generate a matrix
        if (not os.path.exists(
            "../general_parameters/GENERAL_MU_"+str(self.narrowest_layer_size)+".npy")):            # We check if file with mu and sigma2 data to represent foreground exists.
            print("Error: GENERAL PARAMETERS NOT FOUND")
            #print ("Extracting gaussian parameters from encoded images.")
            #self.GENERAL_MU, self.GENERAL_SIGMA2 = util.extract_gaussian_parameters_from_encoded_images(encoder_model_path, encode_training_dataset, L)
            #np.save("../general_parameters/GENERAL_MU_"+str(self.narrowest_layer_size)+".npy", self.GENERAL_MU)
            #np.save("../general_parameters/GENERAL_SIGMA2_"+str(self.narrowest_layer_size)+".npy", self.GENERAL_SIGMA2)
        else:
            print ("Loading guassian parameters from encoded images.")
            #Load data from file to reduce time.
            self.GENERAL_MU = np.load("../general_parameters/GENERAL_MU_"+str(self.narrowest_layer_size)+".npy")
            self.GENERAL_SIGMA2 = np.load("../general_parameters/GENERAL_SIGMA2_"+str(self.narrowest_layer_size)+".npy")

        self.data_initialized = True                                                                                                                                                                                                        # Initialization done.

    def get_initial_background_model_from_training_image_set(self):                                 # Method to get the initial background model.
        print ("Transforming training data to patches.")
        patched_dataset = util.split_in_patches_with_windows_various_images(
            self.normalized_training_dataset, self.patch_height, self.patch_width, 
            self.slicing_window_height, self.slicing_window_width)                                  # We split each image into patches.
        patched_dataset_reshaped = np.reshape(patched_dataset, [patched_dataset.shape[0],
        self.patch_height*self.patch_width*3])                                                      # We reshape them to be one-dimensional patches.
        patched_dataset = None                                                                      # We delete the reference to array in memory to liberate memory.
        self.normalized_training_dataset = None                                                     # We delete the reference to array in memory to liberate memory.
        print ("Encoding initial " + str(
            self.processed_images) + " images to obtain initial gaussian models.")
        predicted_patches = self.encoder.predict(patched_dataset_reshaped)                          # We encode patches
        patched_dataset_reshaped = None                                                             # We delete the reference to array in memory to liberate memory.                                                                                                                                                            #We delete the reference to array in memory to liberate memory.

        print ("Obtaining initial median and standard desviation for each component.")

        for ii in range(self.MU.shape[0]):                                                          # For each patch...
            for jj in range(self.MU.shape[1]):
                absolute_index = ii * self.MU.shape[1] + jj

                for kk in range(self.MU.shape[2]):                                                  # For each component...
                    self.MU[ii, jj, kk] = np.median(
                        predicted_patches[absolute_index::self.patches_per_image, kk])              # We calculate average value
                    #SIGMA2[ii, jj, kk] = np.sum(np.power(
                    #  predicted_patches[absolute_index::patches_per_image, kk] - MU[ii, jj, kk], 2)) / N
                
                self.SIGMA2[ii, jj] = np.copy(self.GENERAL_SIGMA2)                                  # We initialize sigma2 for each patch as foreground sigma2.
                #SIGMA2[ii, jj] = np.array(L*[0.2])
        
        predicted_patches = None                                                                    # We free memory.                                            # We delete the reference to array in memory to liberate memory.
        self.training_done = True                                                                   # We declare training as done.

    def next_image(self, image, training=False):                                                                                    # Method to deal with next image.
        original_image_height_with_padding = self.original_image_height + self.padding_value                                        # We get images sizes with padding.
        original_image_width_with_padding = self.original_image_width + self.padding_value
        if training:                                                                                                                # This image will be used for training.
            if self.training_done:                                                                                                  # If training has been already done.
                print("Error: background model training already done.")                                                             # Error we do nothing.

            else:                                                                                                                   # If training has not be done yet.
                if not self.data_initialized:                                                                                       # If model data has not been initialized
                    self.initialize_data(image)                                                                                     # We initialize them.
                #We normalize the image and put it into matrix to be used to initialize the background model.
                self.normalized_training_dataset[self.processed_images, 
                    (self.padding_value/2):original_image_height_with_padding-(self.padding_value/2), 
                    (self.padding_value/2):original_image_width_with_padding-(self.padding_value/2)] = image.astype( float ) / 255. # We normalize the image and add it to the matrix with padding.         

        else:                                                                                                                       # This image will be used as 
            if not self.training_done:                                                                                              # If training process has not been finalized.
                self.get_initial_background_model_from_training_image_set()                                                         # We apply final calculus over training data and finalize it.
            
            #We add padding and normalize it.
            normalized_image = np.random.uniform(0, 1, 
                [1, original_image_height_with_padding, original_image_width_with_padding, 3])                                      # We generate uniform noise image.
            normalized_image[0, 4:original_image_height_with_padding-4, 
                4:original_image_width_with_padding-4] = image / 255.                                                               # We set normalized image.
            normalized_image = np.reshape(normalized_image, normalized_image.shape[1:])                                             # We delete the first dimension that will always have length 1.
            patched_image = util.split_in_patches_with_windows_one_image(normalized_image, 
                self.patch_height, self.patch_width, self.slicing_window_height, self.slicing_window_width)                         # We split the image into patches.
            normalized_image = None                                                                                                 # We delete the reference to array in memory to liberate memory.  
            patched_image_reshaped = np.reshape(patched_image, [patched_image.shape[0], self.patch_height*self.patch_width*3])      # We reshape it.
            patched_image = None                                                                                                    # We delete the reference to array in memory to liberate memory.  
            predicted_patches = self.encoder.predict(patched_image_reshaped)                                                        # We use the encode the patches.
            patched_image_reshaped = None                                                                                           # We delete the reference to array in memory to liberate memory.  
            self.foreground_matrix_to_show = np.zeros([int(self.original_image_height/self.slicing_window_height),
                int(self.original_image_width/self.slicing_window_width)])                                                          # We initialize foreground matrix as a zero for each patch.

            for ii in range(self.MU.shape[0]):      #For each patch...
                for jj in range(self.MU.shape[1]):
                    absolute_index = ii * self.MU.shape[1] + jj
                    # We need to know patch probability to be foreground.

                    patch_value = predicted_patches[absolute_index]
                    log_prob_v_cond_Fore = -self.narrowest_layer_size/2.*math.log(2*math.pi) - np.sum(np.log(np.sqrt(self.FOREGROUND_SIGMA2))) - 1/2.*np.sum(np.divide(np.power(patch_value - self.FOREGROUND_MU, 2), self.FOREGROUND_SIGMA2))
                    log_prob_v_cond_Back = -self.narrowest_layer_size/2.*math.log(2*math.pi) - np.sum(np.log(np.sqrt(self.SIGMA2[ii,jj]))) - 1/2.*np.sum(np.divide(np.power(patch_value - self.MU[ii, jj], 2), self.SIGMA2[ii, jj]))

                    if (log_prob_v_cond_Back >= log_prob_v_cond_Fore):
                        prob_Back_cond_v = 1./(1 + math.pow(math.e, log_prob_v_cond_Fore - log_prob_v_cond_Back))
                        prob_Fore_cond_v = (1 - prob_Back_cond_v)
                        
                    if (log_prob_v_cond_Back < log_prob_v_cond_Fore):
                        prob_Fore_cond_v = 1./(1 + math.pow(math.e, log_prob_v_cond_Back - log_prob_v_cond_Fore)) 
                        prob_Back_cond_v = (1 - prob_Fore_cond_v)

                    self.foreground_matrix_to_show[ii, jj] += prob_Fore_cond_v
                    self.SIGMA2[ii,jj] = (1 - self.ALPHA * prob_Back_cond_v) * self.SIGMA2[ii,jj] + self.ALPHA * prob_Back_cond_v * np.power(patch_value - self.MU[ii,jj], 2)
                    self.MU[ii,jj] = (1 - self.ALPHA * prob_Back_cond_v) * self.MU[ii,jj] + self.ALPHA * prob_Back_cond_v * patch_value
        self.processed_images += 1
        return self.foreground_matrix_to_show

    def get_foreground_matrix(self):
        return self.foreground_matrix_to_show

    def get_last_segmented_image(self):
        return util.create_segmented_grayscale_image_from_foreground_matrix(self.foreground_matrix_to_show, self.slicing_window_height, self.slicing_window_width, self.original_image_height, self.original_image_width)
