import onnxruntime as ort
import numpy as np
import cv2
import pandas as pd
from torchvision import transforms

# Load class names
class_names = pd.read_csv('data/processed/class_names.csv').values.flatten().tolist()
num_classes = len(class_names)  # Get the number of classes

# Load the ONNX model
onnx_model_path = 'data/models/plant_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

# Define transformation similar to training (normalization, resizing, etc.)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def softmax(logits):
    """Apply softmax to logits to get probabilities."""
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

def predict_disease(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB
    img = cv2.resize(img, (224, 224))  # Ensure image is resized correctly
    img = transform(img)  # Apply transformations
    img = img.unsqueeze(0).numpy()  # Add batch dimension and convert to numpy array

    # Run inference using ONNX Runtime
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)

    # Get raw logits and apply softmax to get probabilities
    logits = ort_outs[0]
    probabilities = softmax(logits)

    # Get the predicted class and its associated confidence
    confidence = np.max(probabilities)
    predicted_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_idx]
    
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    image_path = "data/PlantVillage/Potato___healthy/1dcfeaa9-006d-470c-b3e5-d67609d07d4e___RS_HL 1808.JPG"
    predicted_class, confidence = predict_disease(image_path)
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence}")
