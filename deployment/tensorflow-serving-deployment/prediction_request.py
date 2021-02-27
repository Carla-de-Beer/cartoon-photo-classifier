from tensorflow.keras.preprocessing.image import img_to_array, load_img
import requests

URL = "http://localhost:8501/v1/models/cartoons-photos:predict"

# Load image
img_path = "./unseen/img-03.jpg"
img = load_img(img_path, target_size=(256, 256))
img = img_to_array(img)
img = img / 255.

# Create payload
payload = {"instances": [img.tolist()]}

# Make request
res = requests.post(URL, json=payload)
res = res.json()
print(res)