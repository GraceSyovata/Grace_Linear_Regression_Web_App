from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
try:
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model file not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text='Model not loaded.')
    try:
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)
        prediction = model.predict(final_features)
        return render_template('index.html', prediction_text='Predicted House Price: ${:.2f}'.format(prediction[0]))
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text='Error predicting house price.')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')