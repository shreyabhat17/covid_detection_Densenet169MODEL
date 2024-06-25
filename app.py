from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model

app = Flask(__name__)
model = load_model('covid_classification_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the request
    file = request.files['file']

    # Load and preprocess the image
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Perform the prediction
    prediction = model.predict(img)
    # Add debug print statements
    print(f"Raw prediction: {prediction}")

    predicted_class = 'NON-COVID' if prediction[0][0] > 0.5 else 'COVID'

    # Return the predicted class
    return render_template('result.html', predicted_class=predicted_class)

@app.route('/patient_details')
def patient_details():
    return render_template('patient_details.html')

@app.route('/patient_info', methods=['POST'])
def patient_info():
    name = request.form['name']
    age = request.form['age']
    # Add more variables for other details as needed

    # Process the patient details as required for your project

    return render_template('result.html', predicted_class="Result for Patient: {} - Age: {}".format(name, age))

if __name__ == '__main__':
    app.run()
