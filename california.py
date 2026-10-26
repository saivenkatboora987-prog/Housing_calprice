import streamlit as st
import pandas as pd
import joblib
from streamlit_extras.let_it_rain import rain
import os

# --------------------------------
# 1. Page Configuration
# --------------------------------
st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# --------------------------------
# 2. Load Trained Model & Columns
# --------------------------------
@st.cache_resource
def load_model_and_columns():
    loaded_obj = joblib.load("california.joblib")
    
    # Handle if the file is saved as a dictionary
    if isinstance(loaded_obj, dict):
        model = loaded_obj.get("Model")
        columns = loaded_obj.get("Columns")
        return model, columns
    else:
        # Fallback if it's just the model directly
        # Default columns used for standard California Housing dataset
        default_columns = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms", 
            "Population", "AveOccup", "Latitude", "Longitude"
        ]
        return loaded_obj, default_columns

# --------------------------------
# 3. Header
# --------------------------------
st.title("🏠 California House Price")
st.write(
    "Enter the house details below to estimate "
    "the median house value."
)

st.divider()
# Load model and columns safely upfront
try:
    model, columns = load_model_and_columns()
except FileNotFoundError:
    st.error("Cannot find california.joblib. Check the file location.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --------------------------------
# 4. Input Form
# --------------------------------
st.subheader("📋 House Details")

with st.form("house_prediction_form", border=True):

    values = {}

    # Two-column layout
    col1, col2 = st.columns(2)

    for i, column in enumerate(columns):
        target_col = col1 if i % 2 == 0 else col2

        with target_col:
            values[column] = st.number_input(
                column.replace("_", " ").title(),
                value=0.0,
                format="%0.4f",
                key=column
            )

    submitted = st.form_submit_button(
        "🏠 Predict House Price",
        use_container_width=True,
        type="primary"
    )

# --------------------------------
# 5. Prediction
# --------------------------------
if submitted:
    try:
        # Create input dataframe matching the exact columns
        input_data = pd.DataFrame([[values[c] for c in columns]], columns=columns)
        
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display standard Streamlit metric
        st.metric("🏡 Predicted House Price", f"${prediction:,.2f}")
    except:
        print(f"NA")
# --------------------------------
# 6. Footer
# --------------------------------
st.divider()
st.caption("California House Price Prediction")
st.caption("🏡 Powered by Flask-to-Streamlit conversion and Machine Learning")
rain(
    emoji="🏡",
    font_size=30,
    falling_speed=3,
    animation_length="10" #Duration in seconds
)
