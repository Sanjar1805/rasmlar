from flask import Flask, request, jsonify
import pickle
from PIL import Image
import numpy as np
import io

# Load the model
with open('classificator.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the app
app = Flask(__name__)

# Preprocessing function for images
def preprocess_image(image):
    image = image.resize((224, 224))  # Adjust to your model's input size
    image_array = np.array(image) / 255.0
    image_array = image_array.reshape(1, -1)  # Adjust shape to your model's input
    return image_array

# Serve HTML + CSS + JavaScript in the same response
@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Image Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                margin: 0;
                padding: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
                background: #fff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 400px;
            }
            h1 {
                color: #333;
                margin-bottom: 20px;
            }
            p {
                color: #555;
                margin-bottom: 20px;
            }
            input[type="file"] {
                margin-bottom: 20px;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
            }
            #result.success {
                color: #155724;
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
            }
            #result.error {
                color: #721c24;
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
            }
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Image Classifier</h1>
            <p>Upload an image to classify it:</p>
            <input type="file" id="fileInput" accept="image/*">
            <button id="uploadButton">Upload and Classify</button>
            <div id="result" class="hidden"></div>
        </div>
        <script>
            document.getElementById('uploadButton').addEventListener('click', async () => {
                const fileInput = document.getElementById('fileInput');
                const file = fileInput.files[0];
                const resultDiv = document.getElementById('result');
            
                if (!file) {
                    alert('Please select a file first!');
                    return;
                }
            
                const formData = new FormData();
                formData.append('file', file);
            
                try {
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        body: formData,
                    });
            
                    const data = await response.json();
            
                    if (data.label) {
                        resultDiv.textContent = `Predicted Label: ${data.label}`;
                        resultDiv.className = 'success';
                    } else if (data.error) {
                        resultDiv.textContent = `Error: ${data.error}`;
                        resultDiv.className = 'error';
                    }
                    resultDiv.classList.remove('hidden');
                } catch (err) {
                    resultDiv.textContent = 'An error occurred while uploading the file.';
                    resultDiv.className = 'error';
                    resultDiv.classList.remove('hidden');
                }
            });
        </script>
    </body>
    </html>
    """

@app.route('/api/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_label = prediction[0]
        return jsonify({'label': predicted_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
