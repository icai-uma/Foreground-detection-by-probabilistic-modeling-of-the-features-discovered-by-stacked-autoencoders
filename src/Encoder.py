import keras
import os

class Encoder:                  # Class to represent the encoder model.
    encoder_model = None
    narrowest_layer_size = None

    def __init__(self, encoder_model_path=None, models_path = None, narrowest_layer_size = -1):
        if encoder_model_path == None:
            if models_path != None and narrowest_layer_size != -1:
                models_path = os.path.join(models_path, "models_"+str(narrowest_layer_size), "TinyImage_encoder_model_RGB.h5py")
                print("Loading encoder from " + encoder_model_path)
                self.encoder_model = keras.models.load_model(encoder_model_path)

            else:
                print("Error. No enough information to load the .h5py file.")
        else:
            print("Loading encoder from " + encoder_model_path)
            self.encoder_model = keras.models.load_model(encoder_model_path)
        

    def get_narroest_layer_size(self):
        return self.narrowest_layer_size

    def predict(self,input_data):
        return self.encoder_model.predict(input_data)

