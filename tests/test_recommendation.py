from recommendations import get_recommendations

def test_get_recommendations():
    rec = get_recommendations('Potato___Late_blight')
    assert rec == 'Remove infected leaves immediately and apply copper-based fungicide.'
