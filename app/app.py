from flask import Flask, request, render_template
from inference import predict_disease
from recommendations import get_recommendations
import os

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            UPLOAD_FOLDER = 'uploads'
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)
            file_path = 'uploads/' + file.filename
            file.save(file_path)
            disease, confidence = predict_disease(file_path)
            recommendation = get_recommendations(disease)
            return render_template('result.html', disease=disease, confidence=confidence, recommendation=recommendation)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5001,debug=True)
