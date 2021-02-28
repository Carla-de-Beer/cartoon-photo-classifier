# Cartoon-Photo Classifier: TensorFlow Serving Deployment

TensorFlow Serving acts as a "wrapper" for models by providing an API by which to query models directly. It also provides the infrastructure for hosting a model on a server to allow for production-level scalability. Clients can use HTTP to pass requests to the server along with a data payload. The data is then passed to the model, which will run the inference, get the results, and return them to the client. There are several methods by which TensorFlow Serving can be installed. This example uses Docker.

The model used in this example is the simpler, more lightweight cartoons-photos ```.h5``` model generated [here](https://github.com/Carla-de-Beer/cartoon-photo-classifier/tree/main/classifier).


## Deployment Steps
### Convert the HDF5 binary data format ```.h5``` model to a protobuf ```.pb``` model

TensorFlow Serving operates only with ```.pb``` models. This allows versioning of the models to be used by the clients. Because the latest model version is always used, "model drift", where different clients have different versions of the same model, is prevented. Versioning also allows for the possibility of some clients being issued with a different model version.

Create a folder called ```models```, for example, to work within.

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
There are two options here, either run the docker commands via the command line, or use the Dockerfile provided.

#### Option 1: Command Line Instructions

Your file structure should look like this:

```
|-- models
    |-- cartoons-photos
    |-- 1
        |-- assets
        |-- saved_model.pb
        |-- variables
            |-- variables.data-00000-of-00001
            |-- variables.index
```

Use ```docker pull``` to get the TensorFlow Serving package:

```
docker pull tensorflow/serving
```

Set up a variable called ```MODELDATA``` that contains the path of the sample model:

```
MODELDATA="$(pwd)/models"
```

where ```model``` is the folder containing the ```.pb``` model.

Run TensorFlow Serving from the Docker image:

```
docker run -t --rm -p 8501:8501 \
-v "$MODELDATA/cartoons-photos:/models/cartoons-photos" \
-e MODEL_NAME=cartoons-photos \
tensorflow/serving &
```

#### Option 2: Dockerfile

Using the Dockerfile provided, ensure that the file structure you are using matches the example below:

```
|-- Dockerfile
|-- models
    |-- cartoons-photos
    |-- 1
        |-- assets
        |-- saved_model.pb
        |-- variables
            |-- variables.data-00000-of-00001
            |-- variables.index
```

Run the following docker commands inside the terminal:

```
docker build -t tf-serving-cartoons-photos-classifier .
```
```
docker run -p 8501:8501 tf-serving-cartoons-photos-classifier
```

After following either Option 1 or Option 2, the result will be the instantiation of
a server on port 8501 and the model can then be accessed at:

```
http://localhost:8501/v1/models/cartoons-photos
```

A ```GET``` request to this call should return something like this:

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

A ```POST``` call can then be used to query the API in order to make a request and receive a prediction.

### Query the API

Execute the following Python script:

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
  <img src="unseen/img-04.jpg" width="400px" alt="img-04.jpg"/>
  <figcaption>Fig.1: <code>img-04.jpg</code></figcaption>
</p>

When the request is made again using the unseen image ```img-05.jpg```, the result of this call is

```
{'predictions': [[7.35907233e-05]]}
```

In this case the classifier predicted that the input image is a cartoon with a certainty of (1 - 7.35907233e-05) * 100% = 99.993%. We know that this image is a cartoon and that the prediction is quite accurate.

<br/>
<p align="center">
  <img src="unseen/img-05.jpg" width="400px" alt="img-05.jpg"/>
  <figcaption>Fig.2: <code>img-05.jpg</code></figcaption>
</p>
