import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION & MODERN STYLING
# ==========================================
st.set_page_config(
    page_title="Pune Real Estate Estimator", 
    page_icon="🏢", 
    layout="centered"
)

# Custom CSS for a sleek, modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        width: 100%;
        padding: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #1a252f;
        border-color: #1a252f;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .prediction-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2980b9;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD ASSETS (CACHED FOR PERFORMANCE)
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        model = load_model('model.h5') # Update to model.keras if necessary
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading files: {e}. Ensure model.h5 and scaler.pkl are in the directory.")
        return None, None

model, scaler = load_artifacts()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🏢 Pune Property Value Estimator")
st.markdown("Enter the specifications below to generate a real-time price estimate.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, value=1000.0, step=50.0)
    bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
    bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
    
    # Common Pune locations extracted from your scaler data
    locations = ['Ambegaon Budruk', 'Aundh', 'Baner', 'Bavdhan', 'Deccan Gymkhana', 
                 'Hadapsar', 'Hingne Khurd', 'Kharadi', 'Kothrud', 'Shivaji Nagar', 
                 'Wakadewadi', 'Viman Nagar', 'Other']
    site_location = st.selectbox("Site Location", sorted(locations))
    
    area_types = ['Carpet Area', 'Plot Area', 'Super built-up Area', 'Built-up Area']
    area_type = st.selectbox("Area Type", area_types)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("Estimate Property Price"):
    if model is not None and scaler is not None:
        try:
            # Calculate derived feature exactly as your model expects
            sqft_per_bhk = total_sqft / bhk if bhk > 0 else 0
            
            # Initialize a zero array for all 181 features
            x = np.zeros(181)
            
            # Update base numerical features using indexing
            x[0] = total_sqft
            x[1] = bath
            x[2] = balcony
            x[3] = bhk
            x[4] = sqft_per_bhk
            
            # We fetch the exact expected columns from the scaler to find indices
            # scaler.feature_names_in_ contains the 181 column names used during training
            columns = list(scaler.feature_names_in_)
            
            # Handle Area Type One-Hot Encoding
            area_column = f"area_type_{area_type}"
            if area_column in columns:
                area_index = columns.index(area_column)
                x[area_index] = 1
                
            # Handle Location One-Hot Encoding
            loc_column = f"site_location_{site_location}"
            if loc_column in columns:
                loc_index = columns.index(loc_column)
                x[loc_index] = 1
                
            # Scale the features
            x_scaled = scaler.transform([x])
            
            # Make Prediction
            prediction = model.predict(x_scaled)[0][0]
            
            # Display Result
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Estimated Value</h2>
                    <h1 style="color: #27ae60;">₹ {prediction:,.2f} Lakhs</h1>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
