import tensorflow as tf
import os

MODEL_PATH = "./models/simple_classifier.h5"

MODEL_DIR = os.path.dirname(os.path.dirname(MODEL_PATH))
version = 1

export_path = os.path.join(MODEL_DIR, str(version))
export_path = os.path.join("cartoons-photos", export_path)
print(export_path)

pre_trained_model = tf.keras.models.load_model(MODEL_PATH)
pre_trained_model.save(export_path, save_format="tf")