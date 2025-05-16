import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Configure Streamlit for performance
st.set_page_config(
    page_title="MPG Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed for faster loading
)

# Enable aggressive caching
st.cache_data.clear()
st.cache_resource.clear()

# Define the CustomAttrAdder class (needed for model pipeline)
class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        acc_ix, hpower_ix, cyl_ix = 4, 2, 0
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]

# Function to preprocess origin column - cached for performance
@st.cache_data
def preprocess_origin_cols(df):
    df = df.copy()
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

# Load model and transformer with caching
@st.cache_resource
def load_model_and_transformer():
    try:
        with open('model.bin', 'rb') as f_in:
            model = pickle.load(f_in)
        
        # Try to load the pre-fitted transformer
        try:
            with open('transformer.bin', 'rb') as f_in:
                transformer = pickle.load(f_in)
        except FileNotFoundError:
            # If transformer file doesn't exist, create a basic dataset for fitting
            # This is a fallback solution - ideally you should save the fitted transformer
            df = pd.DataFrame({
                'Cylinders': [4, 6, 8],
                'Displacement': [150.0, 200.0, 350.0],
                'Horsepower': [100.0, 150.0, 200.0],
                'Weight': [2500.0, 3000.0, 4000.0],
                'Acceleration': [15.0, 12.0, 10.0],
                'Model Year': [70, 75, 80],
                'Origin': [1, 2, 3]
            })
            
            # Preprocess origin
            df = preprocess_origin_cols(df)
            
            # Create and fit the transformer
            transformer = ColumnTransformer([
                ("num", Pipeline([
                    ('imputer', SimpleImputer(strategy="median")),
                    ('attrs_adder', CustomAttrAdder()),
                    ('std_scaler', StandardScaler()),
                ]), ['Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year']),
                ("cat", OneHotEncoder(sparse_output=False), ["Origin"]),
            ])
            
            # Fit the transformer
            transformer.fit(df)
            
            # Save the fitted transformer for future use
            try:
                with open('transformer.bin', 'wb') as f_out:
                    pickle.dump(transformer, f_out)
            except Exception as e:
                st.warning(f"Could not save transformer: {e}")
                
        return model, transformer
    except FileNotFoundError:
        return None, None

# Function to make predictions - cached for performance
@st.cache_data
def predict_mpg(_config, _model, _transformer):
    # Convert to DataFrame if dictionary
    if isinstance(_config, dict):
        df = pd.DataFrame(_config)
    else:
        df = _config
    
    # Process data
    df = preprocess_origin_cols(df)
    
    # Transform data
    prepared_df = _transformer.transform(df)
    
    # Make predictions
    return _model.predict(prepared_df)

def main():
    # Immediately load model and transformer in the background
    model, transformer = load_model_and_transformer()
    
    # Header and description - minimal for faster loading
    st.title('MPG Predictor')
    
    if model is None:
        st.error("Model file not found. Please make sure 'model.bin' exists in the current directory.")
        st.stop()
    
    if transformer is None:
        st.error("Transformer could not be initialized. Please check your model and data.")
        st.stop()
    
    # Session state for faster updates
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_value' not in st.session_state:
        st.session_state.prediction_value = 0
    
    # Create a form to batch all inputs together
    with st.form(key='prediction_form'):
        # Two columns for better space utilization
        col1, col2 = st.columns(2)
        
        with col1:
            cylinders = st.selectbox('Cylinders', options=[3, 4, 5, 6, 8], key='cyl')
            displacement = st.slider('Displacement', 50.0, 500.0, 150.0, step=10.0, key='disp')
            horsepower = st.slider('Horsepower', 40.0, 250.0, 100.0, step=5.0, key='hp')
        
        with col2:
            weight = st.slider('Weight (lbs)', 1500.0, 5000.0, 3000.0, step=100.0, key='wt')
            acceleration = st.slider('Acceleration', 8.0, 25.0, 15.0, step=0.5, key='acc')
            model_year = st.select_slider('Model Year', options=list(range(70, 85)), key='year')
            origin = st.selectbox('Origin', options=[1, 2, 3], 
                                format_func=lambda x: {1: "India", 2: "USA", 3: "Germany"}[x],
                                key='origin')
        
        # Submit button for form
        submit_button = st.form_submit_button(label='Predict MPG')
    
    # Only run prediction when form is submitted
    if submit_button:
        with st.spinner('Calculating...'):
            # Create config from inputs
            config = {
                'Cylinders': [cylinders],
                'Displacement': [displacement],
                'Horsepower': [horsepower],
                'Weight': [weight],
                'Acceleration': [acceleration],
                'Model Year': [model_year],
                'Origin': [origin]
            }
            
            # Make prediction
            prediction = predict_mpg(config, model, transformer)
            
            # Save to session state
            st.session_state.prediction_made = True
            st.session_state.prediction_value = prediction[0]
    
    # Display prediction if available
    if st.session_state.prediction_made:
        st.success(f'Estimated MPG: **{st.session_state.prediction_value:.2f}**')
        
        # Efficiency context
        if st.session_state.prediction_value >= 30:
            st.info("This is excellent fuel efficiency for this era!")
        elif st.session_state.prediction_value >= 20:
            st.info("This is good fuel efficiency for this era.")
        else:
            st.info("This fuel efficiency is below average for this era.")
    
    # About section - only expanded if user interacts with it
    with st.expander("About", expanded=False):
        st.write("""
        This app predicts vehicle fuel efficiency (MPG) based on specifications
        using a Random Forest model trained on the Auto MPG dataset.
        """)

if __name__ == "__main__":
    main()