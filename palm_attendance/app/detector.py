import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import json
import os

# Load model config
CONFIG_PATH = "app/config/model_config.json"
with open(CONFIG_PATH) as f:
    config = json.load(f)

MODEL_PATH = config.get("model_path", "models/mobilenet_palm.tflite")
INPUT_SIZE = tuple(config.get("input_size", [224, 224]))
CLASS_NAMES = config.get("class_names")

# If not provided in config, auto-detect from dataset folder
if CLASS_NAMES is None:
    dataset_dir = config.get("dataset_path", "dataset")
    CLASS_NAMES = sorted(os.listdir(dataset_dir)) if os.path.exists(dataset_dir) else ["unknown"]

class PalmVeinRecognizer:
    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, image):
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(image, INPUT_SIZE)
        normalized = resized.astype(np.float32) / 255.0
        input_data = normalized.reshape(1, INPUT_SIZE[0], INPUT_SIZE[1], 1)
        return input_data

    def predict(self, image):
        try:
            processed = self.preprocess(image)
            self.interpreter.set_tensor(self.input_details[0]['index'], processed)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self.output_details[0]['index'])
            prediction = output[0]
            class_idx = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            user_id = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else "unknown"
            return user_id, confidence
        except Exception as e:
            print(f"[Detector Error] {e}")
            return "unknown", 0.0
