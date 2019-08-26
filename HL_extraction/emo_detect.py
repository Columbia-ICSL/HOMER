import numpy as np
from keras.models import model_from_json

class Models(object):
    emotion_model_name='../Models/Emotion/piyush2896_model'
    
    with open(emotion_model_name + ".json", "r") as json_file:
        loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
    model.load_weights(emotion_model_name + ".h5")
    print("Model loaded!")

    


#Class handling the emotion prediction over 7 classes using CNN for one image
class EmoDetector(object):

    EMOTIONS_LIST7 = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Neutral"]

    def __init__(self):
        self.preds7 = []
        self.best_guess7 = []
        

    def predict_emotion(self, img):
        preds = Models.model.predict(img)
        self.preds7.append(np.squeeze(preds))
        self.best_guess7.append(self.EMOTIONS_LIST7[np.argmax(self.preds7[-1])])
        

    def reinitialization(self):
        self.pred7 = self.best_guess7 = []

