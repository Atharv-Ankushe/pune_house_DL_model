import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(page_title="Pune Real Estate Estimator", page_icon="🏢", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2c3e50; color: white; border-radius: 8px;
        width: 100%; padding: 10px; font-weight: bold; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #1a252f; border-color: #1a252f; }
    .prediction-box {
        background-color: #e8f4f8; padding: 20px; border-radius: 10px;
        border-left: 5px solid #2980b9; text-align: center; margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD YOUR SPECIFIC ASSETS
# ==========================================
@st.cache_resource
def load_artifacts():
    try:
        # Loading your specific filenames
        model = load_model('house_price_model.keras')
        
        with open('scaler_X.pkl', 'rb') as f:
            scaler_X = pickle.load(f)
            
        with open('scaler_y.pkl', 'rb') as f:
            scaler_y = pickle.load(f)
            
        with open('model_columns.pkl', 'rb') as f:
            model_columns = pickle.load(f)
            
        return model, scaler_X, scaler_y, model_columns
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

model, scaler_X, scaler_y, model_columns = load_artifacts()

# ==========================================
# 3. USER INTERFACE
# ==========================================
st.title("🏢 Pune Property Value Estimator")
st.divider()

col1, col2 = st.columns(2)

with col1:
    total_sqft = st.number_input("Total Square Feet", min_value=300.0, max_value=10000.0, value=1000.0, step=50.0)
    bhk = st.number_input("Number of Bedrooms (BHK)", min_value=1, max_value=10, value=2)
    bath = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    balcony = st.number_input("Number of Balconies", min_value=0, max_value=5, value=1)
    
    locations = ['Ambegaon Budruk', 'Aundh', 'Baner', 'Deccan Gymkhana', 
                 'Hadapsar', 'Hingne Khurd', 'Kharadi', 'Kothrud', 'Shivaji Nagar', 
                 'Viman Nagar', 'Wakadewadi']
    site_location = st.selectbox("Site Location", sorted(locations))
    
    area_types = ['Carpet Area', 'Plot Area', 'Super built-up Area']
    area_type = st.selectbox("Area Type", area_types)

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button("Estimate Property Price"):
    if model and scaler_X and scaler_y and model_columns:
        try:
            sqft_per_bhk = total_sqft / bhk if bhk > 0 else 0
            
            x = np.zeros(len(model_columns))
            
            x[0] = total_sqft
            x[1] = bath
            x[2] = balcony
            x[3] = bhk
            x[4] = sqft_per_bhk
            
            area_column = f"area_type_{area_type.replace(' ', '  ')}"
            if area_column in model_columns:
                area_index = model_columns.index(area_column)
                x[area_index] = 1
                
            loc_column = f"site_location_{site_location}"
            if loc_column in model_columns:
                loc_index = model_columns.index(loc_column)
                x[loc_index] = 1
                
            x_scaled = scaler_X.transform([x])
            scaled_prediction = model.predict(x_scaled)
            final_price = scaler_y.inverse_transform(scaled_prediction)[0][0]
            
            st.markdown(f"""
                <div class="prediction-box">
                    <h2>Estimated Value</h2>
                    <h1 style="color: #27ae60;">₹ {final_price:,.2f} Lakhs</h1>
                </div>
            """, unsafe_allow_html=True)
            st.balloons()
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")
