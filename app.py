# Import necessary libraries
from flask import Flask, request, jsonify
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Use raw string for file paths
filepath = 'model.h5'
model = load_model(filepath)
print("Model Loaded Successfully")

def pred_tomato_disease(tomato_plant):
    test_image = load_img(tomato_plant, target_size=(128, 128))  # load image 
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # change dimension 3D to 4D

    result = model.predict(test_image)  # predict diseased plant or not
    print('@@ Raw result = ', result)

    pred = np.argmax(result, axis=1)
    print(pred)
    
    disease_map = {
        0: "Tomato - Bacteria Spot Disease",
        1: "Tomato - Early Blight Disease",
        2: "Tomato - Healthy and Fresh",
        3: "Tomato - Late Blight Disease",
        4: "Tomato - Leaf Mold Disease",
        5: "Tomato - Septoria Leaf Spot Disease",
        6: "Tomato - Target Spot Disease",
        7: "Tomato - Tomato Yellow Leaf Curl Virus Disease",
        8: "Tomato - Tomato Mosaic Virus Disease",
        9: "Tomato - Two Spotted Spider Mite Disease"
    }
    
    disease_name = disease_map.get(pred[0], "Unknown Disease")
    return disease_name

# Create flask instance
app = Flask(__name__)

# Render index.html page (not needed for API-only version)
@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Tomato Leaf Disease Prediction API"})

# Get input image from client, predict class, and return JSON response
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # get input
        filename = file.filename        
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/upload', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred = pred_tomato_disease(tomato_plant=file_path)
        
        return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(threaded=False, port=8080)
