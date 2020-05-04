import io
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request

app = Flask(__name__)


def load_model(model_path):
    global model
    model = tf.keras.models.load_model(model_path)


def prepare_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    image = np.reshape(image, (1, 28, 28, 1))
    image = image/255
    return image


def get_prediction(image):
    prediction = model.predict(image)
    prediction = np.argmax(prediction)
    return prediction


@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    image = prepare_image(image)
    prediction = get_prediction(image)
    response = jsonify({"prediction": str(prediction)})
    return response


if __name__ == '__main__':
    load_model('static/models/model1')
    app.run(debug=True)