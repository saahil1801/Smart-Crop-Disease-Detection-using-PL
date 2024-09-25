import torch
from model_training import PlantModel

def test_model_initialization():
    model = PlantModel((3, 224, 224), 3)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 3)
