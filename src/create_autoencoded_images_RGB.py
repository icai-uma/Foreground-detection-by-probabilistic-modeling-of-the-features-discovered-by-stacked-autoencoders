import numpy as np

x_predicted = autoencoder.predict(x_test_with_noise)

x_predicted = np.reshape(x_predicted, [x_predicted.shape[0], 16, 16, 3])

x_predicted = x_predicted * 255.

x_test = np.reshape(x_test, [x_test.shape[0], 16, 16, 3])

x_test_original = np.reshape(x_test_with_noise * 255., [x_test_with_noise.shape[0], 16, 16, 3])

for index in range(x_test_original.shape[0]):
    cv2.imwrite( os.path.join(result_path, 'RGB_img_'+str(index)+'.ori'+'.jpg'), x_test_original[index])
    cv2.imwrite( os.path.join(result_path, 'RGB_img_'+str(index)+'.jpg'), x_predicted[index])


