import numpy as np
import cv2
import os
import util

#First, we need to load all data.
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

x_train_with_noise = util.add_additive_gaussian_noise(x_train.copy(), mu, sigma) #We add gaussian noise.
x_train_with_noise = np.clip(x_train_with_noise, 0, 1, out=x_train_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1

x_train_without_noise = x_train

#print x_train

x_test = testset / 255.

x_test_with_noise = util.add_additive_gaussian_noise(x_test.copy(), mu, sigma) #We add gaussian noise.
x_test_with_noise = np.clip(x_test_with_noise, 0, 1, out=x_test_with_noise) #We ensure that there is no value smaller than 0 and no value greater than 1

x_test_without_noise = x_test

print("Images to train shape:")
print(x_train.shape)

print("Images to test shape:")
print(x_test.shape)
