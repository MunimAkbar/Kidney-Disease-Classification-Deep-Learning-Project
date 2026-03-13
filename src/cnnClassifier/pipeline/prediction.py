import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self._model = None
        self._feature_extractor = None
        self._feature_mean = None
        self._ood_threshold = None

    def _load_model(self):
        """Lazy-load model and OOD reference (only once)."""
        if self._model is not None:
            return

        self._model = load_model(os.path.join("model", "model.h5"))

        # Build feature extractor from flatten layer
        flatten_layer = None
        for layer in self._model.layers:
            if 'flatten' in layer.name.lower():
                flatten_layer = layer
                break
        if flatten_layer is None:
            flatten_layer = self._model.layers[-2]

        self._feature_extractor = Model(
            inputs=self._model.input,
            outputs=flatten_layer.output
        )

        # Load OOD reference files
        mean_path = os.path.join("model", "feature_mean.npy")
        threshold_path = os.path.join("model", "ood_threshold.npy")

        if os.path.exists(mean_path) and os.path.exists(threshold_path):
            self._feature_mean = np.load(mean_path)
            self._ood_threshold = float(np.load(threshold_path)[0])
        else:
            self._feature_mean = None
            self._ood_threshold = None

    @staticmethod
    def _cosine_distance(a, b):
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - (dot / (norm_a * norm_b))

    def predict(self):
        self._load_model()

        # Preprocess image
        test_image = image.load_img(self.filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image) / 255.0
        test_image = np.expand_dims(test_image, axis=0)

        # --- OOD check ---
        if self._feature_mean is not None and self._ood_threshold is not None:
            features = self._feature_extractor.predict(test_image, verbose=0)[0]
            distance = self._cosine_distance(features, self._feature_mean)

            if distance > self._ood_threshold:
                return {
                    "prediction": "Invalid",
                    "confidence": 0,
                    "message": "This image does not appear to be a kidney CT scan.",
                    "distance": round(float(distance), 6),
                    "threshold": round(self._ood_threshold, 6)
                }

        # --- Classification ---
        result = self._model.predict(test_image, verbose=0)
        confidence = float(np.max(result))
        predicted_class = np.argmax(result, axis=1)[0]

        prediction = 'Tumor' if predicted_class == 1 else 'Normal'

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        }