import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
from flask import Flask, request, render_template
import cv2

# Load the trained rice classification model
model = tf.keras.models.load_model(
    filepath='rice.h5',
    custom_objects={'KerasLayer': hub.KerasLayer}
)

app = Flask(__name__, static_folder='static', template_folder='templates')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/details')
def pred():
    return render_template('details.html')


@app.route('/result', methods=['POST'])
def predict():
    try:
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'Data')
        os.makedirs(upload_dir, exist_ok=True)

        filepath = os.path.join(upload_dir, f.filename)
        f.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("Invalid image file")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.array(img, dtype=np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict using the model
        pred = model.predict(img)
        class_names = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']
        prediction = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        return render_template('results.html', prediction_text=prediction, confidence=round(confidence, 2))

    except Exception as e:
        return render_template('error.html', error_message=str(e))


if __name__ == "__main__":
    app.run(debug=True)
