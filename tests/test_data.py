import numpy as np
from data_preprocessing import load_data

def test_load_data():
    images, labels, class_names = load_data('data/PlantVillage', ['Potato___healthy'])
    assert len(images) > 0
    assert len(labels) > 0
    assert len(images) == len(labels)
