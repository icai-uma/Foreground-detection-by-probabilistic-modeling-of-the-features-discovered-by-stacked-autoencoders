import numpy as np
import os
import cv2

def add_additive_gaussian_noise(matrix, mu, sigma, padding_shape=None):
    if (padding_shape is None):
        padding_shape = np.zeros(len(matrix.shape), dtype = np.int8)
    if (mu != 0 or sigma != 0):
        if(len(matrix.shape) >= 3):
            noise_matrix = np.random.normal(mu, sigma, np.subtract(matrix.shape, padding_shape))
            return_matrix = matrix
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2)] += noise_matrix
        elif(len(matrix.shape) == 2):
            noise_matrix = np.random.normal(mu, sigma, [matrix.shape[0] - padding_shape[0], matrix.shape[1] - padding_shape[1]])
            return_matrix = matrix
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2)] += noise_matrix
    else:
        return_matrix = matrix
    
    return return_matrix

def add_additive_uniform_noise(matrix, low = 0.0, high = 1.0, padding_shape = None):
    if (padding_shape is None):
        padding_shape = np.zeros(len(matrix.shape), dtype = np.int8)
    if (low <= high and high != 0.0):
        if(len(matrix.shape) >= 3):
            noise_matrix = np.random.uniform(low, high, np.subtract(matrix.shape, padding_shape))
            return_matrix = matrix
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2)] += noise_matrix
        elif(len(matrix.shape) == 2):
            noise_matrix = np.random.uniform(low, high, [matrix.shape[0] - padding_shape[0], matrix.shape[1] - padding_shape[1]])
            return_matrix = matrix
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2)] += noise_matrix
    else:
        return_matrix = matrix
    
    return return_matrix
    
def add_additive_salt_pepper_noise(matrix, noise_prob, padding_shape = None):
    if (padding_shape is None):
        padding_shape = np.zeros(len(matrix.shape), dtype = np.int8)
        
    if(len(matrix.shape) >= 3):
        return_matrix = matrix
        
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape)[:-1], noise_prob/2)

        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2), i] *=((False==noise_mask) * 1)
            
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape)[:-1], noise_prob/2)
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2), i] += (noise_mask * 1)

        return_matrix = matrix
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape), noise_prob/2)
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2), i] *= ((False==noise_mask) * 1)
            
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape), noise_prob/2)
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2), i] += (noise_mask * 1)
    
    return return_matrix
    
def add_additive_mask_noise(matrix, noise_prob, padding_shape = None):
    if (padding_shape is None):
        padding_shape = np.zeros(len(matrix.shape), dtype = np.int8)
    if(len(matrix.shape) >= 3):
        return_matrix = matrix
        
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape)[:-1], noise_prob)
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2), i] *= ((False==noise_mask) * 1)
    elif(len(matrix.shape) == 2):
        return_matrix = matrix
        
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape), noise_prob)
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2), i] *= ((False==noise_mask) * 1)
    
    return return_matrix
    
def add_additive_mask_noise_by_patches(matrix, noise_prob, patch_size, padding_shape = None):
    if (padding_shape is None):
        padding_shape = np.zeros(len(matrix.shape), dtype = np.int8)
    if(len(matrix.shape) >= 3):
        return_matrix = matrix
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape)[:-1], noise_prob/(patch_size*patch_size))
        original_noise_mask = np.copy(noise_mask)
        for i in range(patch_size):
            for j in range(patch_size):
                if (i > 0 or j > 0):
                    if (i > 0 and j > 0):
                        noise_mask[:,i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[:,0:-i,0:-j]
                    elif (j > 0):
                        noise_mask[:,i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[:,:,0:-j]
                    elif (i > 0):
                        noise_mask[:,i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[:,0:-i,:]
        
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):(matrix.shape[0]-padding_shape[0]/2), (padding_shape[1]/2):(matrix.shape[1]-padding_shape[1]/2), (padding_shape[2]/2):(matrix.shape[2]-padding_shape[2]/2), i] *= ((False==noise_mask) * 1)
    elif(len(matrix.shape) == 2):
        return_matrix = matrix
        noise_mask = generate_noise_mask(np.subtract(matrix.shape, padding_shape), noise_prob/patch_size)
        for i in range(patch_size):
            for j in range(patch_size):
                if (i > 0 or j > 0):
                    noise_mask[i:,j:] = noise_mask[i:,j:]|noise_mask[:-i,:-j]
                    if (i > 0 and j > 0):
                        noise_mask[i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[0:-i,0:-j]
                    elif (j > 0):
                        noise_mask[i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[:,0:-j]
                    elif (i > 0):
                        noise_mask[i:,j:] = noise_mask[:,i:,j:]|original_noise_mask[0:-i,:]
                    
        for i in range(matrix.shape[-1]):
            return_matrix[(padding_shape[0]/2):matrix.shape[0]-(padding_shape[0]/2), (padding_shape[1]/2):matrix.shape[1]-(padding_shape[1]/2), i] *= ((False==noise_mask) * 1)
    
    return return_matrix
    
def generate_noise_mask(matrix_shape, noise_probability):
    uniform_noise_matrix = np.random.uniform(0., 1., matrix_shape)
    noise_mask = uniform_noise_matrix <= noise_probability
    return noise_mask
    

def generate_image_dataset_with_Gaussian_noise(original_images_folder, final_images_folder, noise_mu, noise_sigma):
    if os.path.exists(original_images_folder):

        if not os.path.exists(final_images_folder): #If objective folder does not exist, we create it.
            os.makedirs(final_images_folder)

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        normalized_dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            normalized_dataset[img_index] = img.astype( float ) / 255.
            img_index += 1

        #We add the noise.
        dataset_with_noise = add_additive_gaussian_noise(normalized_dataset, noise_mu, noise_sigma)
        dataset_with_noise = np.clip(dataset_with_noise, 0, 1, out = dataset_with_noise)

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(final_images_folder, img_name)
            img = dataset_with_noise[img_index] * 255.
            img = img.astype(int)
            cv2.imwrite( img_dir, img)
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue

def generate_image_dataset_with_uniform_noise(original_images_folder, final_images_folder, low = 0.0, high = 1.0):
    if os.path.exists(original_images_folder):

        if not os.path.exists(final_images_folder): #If objective folder does not exist, we create it.
            os.makedirs(final_images_folder)

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        normalized_dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            normalized_dataset[img_index] = img.astype( float ) / 255.
            img_index += 1

        #We add the noise.
        dataset_with_noise = add_additive_uniform_noise(normalized_dataset, low, high)
        dataset_with_noise = np.clip(dataset_with_noise, 0, 1, out = dataset_with_noise)

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(final_images_folder, img_name)
            img = dataset_with_noise[img_index] * 255.
            img = img.astype(int)
            cv2.imwrite( img_dir, img)
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue
    
def generate_image_dataset_with_salt_pepper_noise(original_images_folder, final_images_folder, noise_prob):
    if os.path.exists(original_images_folder):

        if not os.path.exists(final_images_folder): #If objective folder does not exist, we create it.
            os.makedirs(final_images_folder)

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        normalized_dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            normalized_dataset[img_index] = img.astype( float ) / 255.
            img_index += 1

        #We add the noise.
        dataset_with_noise = add_additive_salt_pepper_noise(normalized_dataset, noise_prob)
        dataset_with_noise = np.clip(dataset_with_noise, 0, 1, out = dataset_with_noise)

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(final_images_folder, img_name)
            img = dataset_with_noise[img_index] * 255.
            img = img.astype(int)
            cv2.imwrite( img_dir, img)
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue
    
def generate_image_dataset_with_mask_noise(original_images_folder, final_images_folder, noise_prob):
    if os.path.exists(original_images_folder):

        if not os.path.exists(final_images_folder): #If objective folder does not exist, we create it.
            os.makedirs(final_images_folder)

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        normalized_dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            normalized_dataset[img_index] = img.astype( float ) / 255.
            img_index += 1

        #We add the noise.
        dataset_with_noise = add_additive_mask_noise(normalized_dataset, noise_prob)
        dataset_with_noise = np.clip(dataset_with_noise, 0, 1, out = dataset_with_noise)

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(final_images_folder, img_name)
            img = dataset_with_noise[img_index] * 255.
            img = img.astype(int)
            cv2.imwrite( img_dir, img)
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue
    
def generate_image_dataset_with_mask_noise_by_patches(original_images_folder, final_images_folder, noise_prob, patch_size):
    if os.path.exists(original_images_folder):

        if not os.path.exists(final_images_folder): #If objective folder does not exist, we create it.
            os.makedirs(final_images_folder)

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        normalized_dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            normalized_dataset[img_index] = img.astype( float ) / 255.
            img_index += 1

        #We add the noise.
        dataset_with_noise = add_additive_mask_noise_by_patches(normalized_dataset, noise_prob, patch_size)
        dataset_with_noise = np.clip(dataset_with_noise, 0, 1, out = dataset_with_noise)

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(final_images_folder, img_name)
            img = dataset_with_noise[img_index] * 255.
            img = img.astype(int)
            cv2.imwrite( img_dir, img)
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue
    
def generate_image_dataset_with_compression_noise(original_images_folder, final_images_folder, compression_quality):
    if os.path.exists(original_images_folder):

        if not os.path.exists(os.path.join(final_images_folder,"input")): #If objective folder does not exist, we create it.
            os.makedirs(os.path.join(final_images_folder,"input"))

        dataset_dirs = os.listdir(original_images_folder)
        dataset_dirs = sorted(dataset_dirs)
        original_dataset_length = len(dataset_dirs)
        #We get an image from the dataset to obtain images height and width.
        if original_dataset_length > 0:
            img_name = dataset_dirs[0]
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            original_image_height = img.shape[0]
            original_image_width = img.shape[1]

        dataset = np.zeros([original_dataset_length, original_image_height, original_image_width, 3])

        img_index = 0
        for img_name in dataset_dirs:
            img_dir = os.path.join(original_images_folder, img_name)
            img = cv2.imread( img_dir )
            dataset[img_index] = img
            img_index += 1

        #We add the noise while saving images.
        
        img_index = 0
        for img_name in dataset_dirs:
            img = dataset[img_index]
            [name, ext] = os.path.splitext(img_name)
            img_dir = os.path.join(final_images_folder,"input", name + '.jpg')
            print(img_dir)
            cv2.imwrite(img_dir, img, [cv2.IMWRITE_JPEG_QUALITY, compression_quality])
            img_index += 1

        returnValue = True
    else:
        returnValue = False #It is impossible to create dataset with noise due to not folder existence.

    return returnValue
