import pandas as pd

# Example recommendations database
RECOMMENDATIONS = {
    'Potato___Late_blight': 'Remove infected leaves immediately and apply copper-based fungicide.',
    'Potato___Early_blight': 'Use fungicide Z and rotate crops to prevent soil contamination.',
    'Potato___healthy': 'No action required. Maintain regular monitoring and proper plant care.'
}

def get_recommendations(disease_name):
    return RECOMMENDATIONS.get(disease_name, "No recommendations available for this disease.")