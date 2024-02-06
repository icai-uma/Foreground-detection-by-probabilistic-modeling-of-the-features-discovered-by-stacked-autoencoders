from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils import plot_model
from keras.constraints import max_norm, min_max_norm
from keras import regularizers
import os
## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
import numpy as np
import cv2
import util                                 # pylint: disable=import-error 
import noise_util

def load_tiny_images_data(trainset_dir, testset_dir, mu, sigma):

    print ("Loading training and test data.")

    trainset_dirs = os.listdir(trainset_dir)
    testset_dirs = os.listdir(testset_dir)

    trainset_size = len(trainset_dirs)
    trainset = np.zeros([trainset_size * 4, 16, 16, 3])
    index = 0
    for img_name in trainset_dirs:
        img_dir = os.path.join(trainset_dir, img_name)
        #print(img_dir)
        img = cv2.imread( img_dir )
        trainset[index] = img.astype( float )[0:16,0:16]
        index += 1
        trainset[index] = img.astype( float )[16:32,0:16]
        index += 1
        trainset[index] = img.astype( float )[0:16,16:32]
        index += 1
        trainset[index] = img.astype( float )[16:32,16:32]
        index += 1

    trainset = np.reshape(trainset, [trainset_size * 4, 16*16*3])

    testset_size = len(testset_dirs)
    testset = np.zeros([testset_size * 4, 16, 16, 3])
    index = 0
    for img_name in testset_dirs:
        img_dir = os.path.join(testset_dir, img_name)
        #print(img_dir)
        img = cv2.imread( img_dir )
        testset[index] = img.astype( float )[0:16,0:16]
        index += 1
        testset[index] = img.astype( float )[16:32,0:16]
        index += 1
        testset[index] = img.astype( float )[0:16,16:32]
        index += 1
        testset[index] = img.astype( float )[16:32,16:32]
        index += 1

    testset = np.reshape(testset, [testset_size * 4, 16*16*3])

    x_train = trainset / 255.

    x_train_with_noise = noise_util.add_additive_gaussian_noise(x_train.copy(), mu, sigma) #We add gaussian noise.
    x_train_with_noise = np.clip(x_train_with_noise, 0, 1, out=x_train_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1

    x_train_without_noise = x_train

    #print x_train

    x_test = testset / 255.

    x_test_with_noise = noise_util.add_additive_gaussian_noise(x_test.copy(), mu, sigma) #We add gaussian noise.
    x_test_with_noise = np.clip(x_test_with_noise, 0, 1, out=x_test_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1

    x_test_without_noise = x_test

    print("Images to train shape:")
    print(x_train.shape)

    print("Images to test shape:")
    print(x_test.shape)

    return x_train_without_noise, x_train_with_noise, x_test_without_noise, x_test_with_noise


def train_denoising_autoencoder_with_16_as_narrowest_layer(train_set_original, train_set_with_noise, test_set_original, test_set_with_noise, train_epochs, train_batch_size, result_model_path):

    if not os.path.exists(result_model_path):
        os.makedirs( result_model_path )

    if not os.path.exists(result_model_path):
        os.makedirs( result_model_path )

    #TRAINING PREPARATION

    image_channels = 1

    input_img = Input(shape=(16*16*3,))

    #NETWORK DEFINITION

    print("Encoder layer architecture:")
    print(input_img.shape)
    x = Dense(512, activation = 'relu')(input_img)
    print(x.shape)
    x = Dense(256, activation = 'relu')(x)
    print(x.shape)
    x = Dense(128, activation = 'relu')(x)
    print(x.shape)
    x = Dense(64, activation = 'relu')(x)
    print(x.shape)
    x = Dense(32, activation = 'relu')(x)
    print(x.shape)
    encoder = Dense(16, activation = 'sigmoid', name = "encoder")(x)

    print("Encoder output shape:")
    print(encoder.shape)

    print("Decoder layer architecture:")

    x = Dense(32, activation = 'relu')(encoder)
    print(x.shape)
    x = Dense(64, activation = 'relu')(x)
    print(x.shape)
    x = Dense(128, activation = 'relu')(x)
    print(x.shape)
    x = Dense(256, activation = 'relu')(x)
    print(x.shape)
    x = Dense(512, activation = 'relu')(x)
    print(x.shape)
    decoder = Dense(16*16*3, activation = 'sigmoid', name = "decoder")(x)

    print("Decoder output shape:")
    print(decoder.shape)

    autoencoder = Model(input_img, decoder)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

    #NETWORK TRAINING

    print("Train array shape:")
    print(train_set_original.shape)
    print("Test array shape:")
    print(test_set_original.shape)

    autoencoder.fit(train_set_with_noise, train_set_original,
                    epochs=train_epochs,
                    batch_size=train_batch_size,
                    shuffle=True,
                    validation_data=(test_set_with_noise, test_set_original))

    print("Autoencoder Input and Output:")
    print(autoencoder.input)
    print(autoencoder.output)

    plot_model(autoencoder, to_file = result_model_path + "_autoencoder_model_RGB.png", show_shapes = True)

    decoder_input_shape = Input(shape=(16,))

    print(decoder_input_shape)

    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)

    plot_model(encoder, to_file = result_model_path + "_encoder_model_RGB.png", show_shapes = True)
        
    deco = autoencoder.layers[-6](decoder_input_shape)
    deco = autoencoder.layers[-5](deco)
    deco = autoencoder.layers[-4](deco)
    deco = autoencoder.layers[-3](deco)
    deco = autoencoder.layers[-2](deco)
    deco = autoencoder.layers[-1](deco)
    # create the decoder model
    decoder = Model(decoder_input_shape, deco)

    plot_model(decoder, to_file = result_model_path + "_decoder_model_RGB.png", show_shapes = True)

    print("Saving encoder_model")
    encoder.save(result_model_path + "_encoder_model_RGB.h5py")

    print("Saving decoder_model")
    decoder.save(result_model_path + "_decoder_model_RGB.h5py")

    print("Saving autoencoder model")
    autoencoder.save(result_model_path + "_autoencoder_model_RGB.h5py")
