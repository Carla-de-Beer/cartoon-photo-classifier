# Cartoon-Photo Classifier: TensorFlow-Serving Deployment


## Deployment Steps
### Convert the .h5 model to a .pb model

```
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
```

### Run the Docker Container

```
TESTDATA="$(pwd)/testdata"
```

```
docker run -t --rm -p 8501:8501 \
-v "$TESTDATA/cartoons-photos:/models/cartoons-photos" \
-e MODEL_NAME=cartoons-photos \
tensorflow/serving &
```

The model can be reached at the following API URL:

```
http://localhost:8501/v1/models/cartoons-photos
```

and should return something like this:

```
{
    "model_version_status": [
        {
            "version": "1",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```

We now need to make a POST call in order to receive a prediction.

### Call the API URL
```
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests

URL = "http://localhost:8501/v1/models/cartoons-photos:predict"

# Load image
img_path = './unseen/img-04.jpg'
img = load_img(img_path, target_size=(256, 256))
img = img_to_array(img)
img = img / 255.

# Create payload
payload = {"instances": [img.tolist()]}

# Make request
res = requests.post(URL, json=payload)
res = res.json()
print(res)
```

The result of this call is

```
{'predictions': [[0.675242066]]}
```

Cartoons are classified as being close to 0, and photos as close to 1. In this case the model predicted the input image, we know to be a photo, with 67.542% certainty as being a photo.

<br/>
<p align="center">
  <img src="unseen/img-04.jpg" width="400px" alt="joinplot_cartoons"/>
  <figcaption>Fig.1: <code>img-04.jpg</code></figcaption>
</p>

When the request is made again using the unseen image "img-05.jpg", result of this call is

```
{'predictions': [[7.35907233e-05]]}
```

In this case the classifier predicted that the input image is a cartoon with a certainty of (1 - 7.35907233e-05) * 100% = 99.993%. We know that this image is a cartoon and that the prediction is quite accurate. Pretty good.

<br/>
<p align="center">
  <img src="unseen/img-05.jpg" width="400px" alt="joinplot_cartoons"/>
  <figcaption>Fig.2: <code>img-05.jpg</code></figcaption>
</p>
