import time
import datetimeç
import os
from keras import backend as k

import training_util

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

start_time = time.time()

trainset_dir = "../../data/tiny_images/tinyOutput3/train"
testset_dir = "../../data/tiny_images/tinyOutput3/test"

model_path = "../network_models/models30"

output_name = "Ner_patches_TinyImage"

train_epochs = 5000
train_batch_size = 100

mu, sigma = 0, 0.1 #Training gaussian noise parameters.

x_train_original, x_train_with_noise, x_test_original, x_test_with_noise = training_util.load_tiny_images_data(trainset_dir, testset_dir, mu, sigma)

training_util.train_denoising_autoencoder_with_16_as_narrowest_layer(x_train_original, x_train_with_noise, x_test_original, x_test_with_noise, train_epochs, train_batch_size, model_path)


end_time = time.time()

print (str(datetime.timedelta(seconds=(end_time-start_time))))

quit()