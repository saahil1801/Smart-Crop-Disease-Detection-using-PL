from inference import predict_disease

def test_predict_disease():
    image_path = 'data/PlantVillage/Potato___Early_blight/0a8a68ee-f587-4dea-beec-79d02e7d3fa4___RS_Early.B 8461.JPG'
    predicted_class, confidence = predict_disease(image_path)
    assert predicted_class in ['Potato___Late_blight', 'Potato___Early_blight', 'Potato___healthy']
    assert 0 <= confidence <= 1
