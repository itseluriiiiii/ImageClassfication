import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("wildlife_model.h5")

# Class names
class_names = ['cat', 'dog', 'elephant']

# Load image
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

# Predict
prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]

print("Predicted Animal:", predicted_class)