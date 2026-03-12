import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    
    def predict(self):
        # load model from the training artifacts
        model = load_model(os.path.join("model", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0  # normalize like training
        test_image = np.expand_dims(test_image, axis=0)
        
        result = model.predict(test_image)
        confidence = float(np.max(result))
        predicted_class = np.argmax(result, axis=1)[0]

        if predicted_class == 1:
            prediction = 'Tumor'
        else:
            prediction = 'Normal'

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        }