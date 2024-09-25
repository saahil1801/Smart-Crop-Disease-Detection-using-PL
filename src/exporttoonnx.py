import torch
import numpy as np
from model_training import PlantModel
import pytorch_lightning as pl
import os

# Dummy input based on your model's input size (adjust accordingly)
dummy_input = torch.randn(1, 3, 224, 224)
export_dir = 'data/models'


# Define input shape and number of classes
input_shape = (3, 224, 224)
num_classes = 3  # Number of classes in your case

# Load the PyTorch Lightning checkpoint
checkpoint = torch.load('data/models/best_model.ckpt')

# Initialize the model
model = PlantModel(input_shape=input_shape, num_classes=num_classes)

# Load only the state_dict of the model from the Lightning checkpoint
model.load_state_dict(checkpoint['state_dict'])  # Extract the model state dict

model.eval()  # Set the model to evaluation mode


onnx_path = os.path.join(export_dir, "plant_model.onnx")


# Export the model to an ONNX file
torch.onnx.export(
    model,                         # The model to export
    dummy_input,                   # A dummy input (same shape as real input)
    onnx_path,           # The name of the output ONNX file
    export_params=True,            # Store the trained parameters
    opset_version=11,              # ONNX opset version to use (default 11)
    do_constant_folding=True,      # Optimize model by folding constants
    input_names=['input'],         # Name the input node
    output_names=['output'],       # Name the output node
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Dynamic batching
)

print("Model exported to plant_model.onnx")
