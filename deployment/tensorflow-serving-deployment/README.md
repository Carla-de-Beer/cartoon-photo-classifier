# Cartoon-Photo Classifier: TensorFlow Serving Deployment

TensorFlow Serving acts as a simple “wrapper” for models that provides an API surface as well as production-level scalability. It provides the infrastructure for hosting a model on a server. Clients can then use HTTP to pass requests to the server along with a data payload. The data will be passed to the model, which will run inference, get the results, and return them to the client. There are several methods by which you can install TensorFlow Serving. This example uses Docker.

The model used in this example is the simpler, more lightweight cartoons-photos ```.h5``` model generated [here](https://github.com/Carla-de-Beer/cartoon-photo-classifier/tree/main/classifier).


## Deployment Steps
### Convert the ```.h5``` model to a ```.pb``` model

TensorFlow Serving operates only with ```.pb``` models. This allows versioning of the models to be used by the clients. Because the latest version is always used, it prevents "model drift", where different clients have different versions of the same model. This allows allows for the possibility of some clients being issued with a different model version.

Start by converting the existing ```.h5``` model to a ```.pb``` model:

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

Use docker pull to get the TensorFlow Serving package:

```
docker pull tensorflow/serving
```

Set up a variable called ```TESTDATA``` that contains the path of the sample model:

```
TESTDATA="$(pwd)/testdata"
```

Run TensorFlow Serving from the Docker image:

```
docker run -t --rm -p 8501:8501 \
-v "$TESTDATA/cartoons-photos:/models/cartoons-photos" \
-e MODEL_NAME=cartoons-photos \
tensorflow/serving &
```

This will instantiate a server on port 8501 and the model can then be accessed at:

```
http://localhost:8501/v1/models/cartoons-photos
```

The ```GET``` response to this call should return something like this:

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

A ```POST``` call can then be used in order to make a request and receive a prediction.

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

Cartoons are expected to return a result close to 0, and photos close to 1. In this case the model predicted the input image, we know to be a photo, as a photo with 67.542% certainty.

<br/>
<p align="center">
  <img src="unseen/img-04.jpg" width="400px" alt="joinplot_cartoons"/>
  <figcaption>Fig.1: <code>img-04.jpg</code></figcaption>
</p>

When the request is made again using the unseen image ```img-05.jpg```, the result of this call is

```
{'predictions': [[7.35907233e-05]]}
```

In this case the classifier predicted that the input image is a cartoon with a certainty of (1 - 7.35907233e-05) * 100% = 99.993%. We know that this image is a cartoon and that the prediction is quite accurate.

<br/>
<p align="center">
  <img src="unseen/img-05.jpg" width="400px" alt="joinplot_cartoons"/>
  <figcaption>Fig.2: <code>img-05.jpg</code></figcaption>
</p>
