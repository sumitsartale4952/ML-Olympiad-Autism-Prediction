from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Function to predict
def predict(model, input_data):
    # Example: Assume model.predict() takes a DataFrame and returns predictions
    predictions = model.predict(input_data)
    return predictions

# Endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict_autism():
    try:
        # Load model
        model = load_model('best_model.pkl')  # Replace with your actual model file path

        # Get data from request
        data = request.get_json()

        # Prepare input data
        input_data = pd.DataFrame(data)

        # Make prediction
        prediction = predict(model, input_data)

        # Return prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})  # Assuming prediction is a list or single value
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
