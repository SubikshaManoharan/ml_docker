import streamlit as st
import numpy as np
import pickle
from pathlib import Path

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / 'brest_cancer.pkl'
    if not model_path.exists():
        st.error("Model file 'brest_cancer.pkl' not found. Ensure it's in the project root and committed to the repo.")
        st.stop()
    with model_path.open('rb') as f:
        return pickle.load(f)

# Load the trained model once and cache it
model = load_model()

# Detect model classes (safe fallback if attribute missing)
try:
    model_classes = list(model.classes_)
except Exception:
    model_classes = [0, 1]

# Default assumption: numeric label 1 = Malignant when present
default_malignant_value = 1 if 1 in model_classes else model_classes[-1]

# Allow the user to choose which class value corresponds to Malignant
selected_malignant = st.sidebar.selectbox(
    'Select class value that represents Malignant (if unsure, try different values)',
    options=model_classes,
    index=model_classes.index(default_malignant_value) if default_malignant_value in model_classes else 0,
    format_func=lambda x: str(x)
)

# Title
st.title('🔬 Breast Cancer Prediction App')
st.write("This app predicts whether a tumor is **Malignant** or **Benign** based on cell nuclei features.")

# Initialize session state for caching
if 'input_features' not in st.session_state:
    st.session_state.input_features = None
    st.session_state.prediction = None
    st.session_state.prediction_proba = None

def get_default_values():
    return {
        'mean_radius': 14.0,
        'mean_texture': 20.0,
        'mean_perimeter': 90.0,
        'mean_area': 500.0,
        'mean_smoothness': 0.1,
        'mean_compactness': 0.1,
        'mean_concavity': 0.1,
        'mean_concave_points': 0.05,
        'mean_symmetry': 0.2,
        'mean_fractal_dimension': 0.06,
        'radius_error': 1.0,
        'texture_error': 1.5,
        'perimeter_error': 5.0,
        'area_error': 40.0,
        'smoothness_error': 0.01,
        'compactness_error': 0.02,
        'concavity_error': 0.02,
        'concave_points_error': 0.01,
        'symmetry_error': 0.02,
        'fractal_dimension_error': 0.01,
        'worst_radius': 16.0,
        'worst_texture': 25.0
    }

def update_features():
    # Only update if inputs have changed
    current_values = {
        'mean_radius': st.session_state.mean_radius,
        'mean_texture': st.session_state.mean_texture,
        'mean_perimeter': st.session_state.mean_perimeter,
        'mean_area': st.session_state.mean_area,
        'mean_smoothness': st.session_state.mean_smoothness,
        'mean_compactness': st.session_state.mean_compactness,
        'mean_concavity': st.session_state.mean_concavity,
        'mean_concave_points': st.session_state.mean_concave_points,
        'mean_symmetry': st.session_state.mean_symmetry,
        'mean_fractal_dimension': st.session_state.mean_fractal_dimension,
        'radius_error': st.session_state.radius_error,
        'texture_error': st.session_state.texture_error,
        'perimeter_error': st.session_state.perimeter_error,
        'area_error': st.session_state.area_error,
        'smoothness_error': st.session_state.smoothness_error,
        'compactness_error': st.session_state.compactness_error,
        'concavity_error': st.session_state.concavity_error,
        'concave_points_error': st.session_state.concave_points_error,
        'symmetry_error': st.session_state.symmetry_error,
        'fractal_dimension_error': st.session_state.fractal_dimension_error,
        'worst_radius': st.session_state.worst_radius,
        'worst_texture': st.session_state.worst_texture
    }
    
    if st.session_state.input_features is None or any(
        st.session_state.input_features[0][i] != val 
        for i, val in enumerate(current_values.values())
    ):
        st.session_state.input_features = np.array(list(current_values.values())).reshape(1, -1)
        # Clear previous predictions when inputs change
        st.session_state.prediction = None
        st.session_state.prediction_proba = None

# Sidebar input
st.sidebar.header('🧬 Input Tumor Characteristics')

# Set default values
defaults = get_default_values()

# Create input fields with unique keys and default values
st.sidebar.number_input('Mean Radius', 6.0, 30.0, key='mean_radius', on_change=update_features, value=defaults['mean_radius'])
st.sidebar.number_input('Mean Texture', 9.0, 40.0, key='mean_texture', on_change=update_features, value=defaults['mean_texture'])
st.sidebar.number_input('Mean Perimeter', 40.0, 190.0, key='mean_perimeter', on_change=update_features, value=defaults['mean_perimeter'])
st.sidebar.number_input('Mean Area', 150.0, 2500.0, key='mean_area', on_change=update_features, value=defaults['mean_area'])
st.sidebar.number_input('Mean Smoothness', 0.05, 0.16, key='mean_smoothness', on_change=update_features, value=defaults['mean_smoothness'])
st.sidebar.number_input('Mean Compactness', 0.02, 0.35, key='mean_compactness', on_change=update_features, value=defaults['mean_compactness'])
st.sidebar.number_input('Mean Concavity', 0.0, 0.45, key='mean_concavity', on_change=update_features, value=defaults['mean_concavity'])
st.sidebar.number_input('Mean Concave Points', 0.0, 0.2, key='mean_concave_points', on_change=update_features, value=defaults['mean_concave_points'])
st.sidebar.number_input('Mean Symmetry', 0.1, 0.3, key='mean_symmetry', on_change=update_features, value=defaults['mean_symmetry'])
st.sidebar.number_input('Mean Fractal Dimension', 0.04, 0.1, key='mean_fractal_dimension', on_change=update_features, value=defaults['mean_fractal_dimension'])

st.sidebar.number_input('Radius Error', 0.1, 3.0, key='radius_error', on_change=update_features, value=defaults['radius_error'])
st.sidebar.number_input('Texture Error', 0.3, 5.0, key='texture_error', on_change=update_features, value=defaults['texture_error'])
st.sidebar.number_input('Perimeter Error', 1.0, 30.0, key='perimeter_error', on_change=update_features, value=defaults['perimeter_error'])
st.sidebar.number_input('Area Error', 6.0, 550.0, key='area_error', on_change=update_features, value=defaults['area_error'])
st.sidebar.number_input('Smoothness Error', 0.002, 0.03, key='smoothness_error', on_change=update_features, value=defaults['smoothness_error'])
st.sidebar.number_input('Compactness Error', 0.002, 0.15, key='compactness_error', on_change=update_features, value=defaults['compactness_error'])
st.sidebar.number_input('Concavity Error', 0.0, 0.4, key='concavity_error', on_change=update_features, value=defaults['concavity_error'])
st.sidebar.number_input('Concave Points Error', 0.0, 0.05, key='concave_points_error', on_change=update_features, value=defaults['concave_points_error'])
st.sidebar.number_input('Symmetry Error', 0.007, 0.08, key='symmetry_error', on_change=update_features, value=defaults['symmetry_error'])
st.sidebar.number_input('Fractal Dimension Error', 0.001, 0.03, key='fractal_dimension_error', on_change=update_features, value=defaults['fractal_dimension_error'])

st.sidebar.number_input('Worst Radius', 7.0, 40.0, key='worst_radius', on_change=update_features, value=defaults['worst_radius'])
st.sidebar.number_input('Worst Texture', 12.0, 50.0, key='worst_texture', on_change=update_features, value=defaults['worst_texture'])

# Initialize input features if not already done
if st.session_state.input_features is None:
    st.session_state.input_features = np.array(list(defaults.values())).reshape(1, -1)

# Predict button with caching
if st.button('Predict'):
    with st.spinner('Making prediction...'):
        if st.session_state.prediction is None:
            st.session_state.prediction = model.predict(st.session_state.input_features)
            st.session_state.prediction_proba = model.predict_proba(st.session_state.input_features)
        
        # Map prediction to user-selected malignant class value
        pred_value = st.session_state.prediction[0]
        # find index of malignant class in model.classes_
        try:
            malignant_index = model_classes.index(selected_malignant)
        except ValueError:
            malignant_index = 1 if len(model_classes) > 1 else 0

        # Confidence for the malignant class (use the correct column)
        conf_malignant = float(st.session_state.prediction_proba[0][malignant_index]) if st.session_state.prediction_proba is not None else None

        if pred_value == selected_malignant:
            conf_text = f"🔵 Confidence: {round(conf_malignant*100,2)}%" if conf_malignant is not None else ""
            st.error(f'🚨 Prediction: Malignant Tumor\n\n{conf_text}')
        else:
            # Confidence for benign is 1 - malignant confidence when binary
            conf_benign = None
            if conf_malignant is not None and len(model_classes) == 2:
                conf_benign = 1.0 - conf_malignant
            elif st.session_state.prediction_proba is not None:
                # find index of predicted class
                try:
                    pred_index = model_classes.index(pred_value)
                    conf_benign = float(st.session_state.prediction_proba[0][pred_index])
                except Exception:
                    conf_benign = None

            conf_text = f"🟢 Confidence: {round(conf_benign*100,2)}%" if conf_benign is not None else ""
            st.success(f'✅ Prediction: Benign Tumor\n\n{conf_text}')

st.markdown("---")
st.caption("Built with ❤️ ")