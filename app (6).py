import streamlit as st
import pandas as pd
import joblib

# 1. Page Styling
st.set_page_config(page_title="AI-Bridge Sales Predictor", layout="centered")

st.title("Patients Details")
st.write("Enter the result details to predict healthy status.")

try:
    # 2. Load the trained model
    model = joblib.load('/content/Patients_Details.pkl')

    # 3. Create a Layout with Columns for User Input
    col1, col2, col3 = st.columns(3)

    with col1:
        glucose = st.number_input("Glucose", min_value=44.0, max_value=500.0, value=44.0)

    with col2:
        bp = st.number_input("Bloodpressure", min_value=24.0, max_value=100.0, value=24.0)

    with col3:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=200.0, value=0.0)

    # 4. Create a 'Predict' button
    if st.button("Calculate Prediction"):
        # Create a DataFrame from the dynamic user input
        user_input = pd.DataFrame([{
            'Glucose': glucose,
            'Bloodpressure': bp,
            'Insulin': insulin
        }])

        # Get prediction
        prediction = model.predict(user_input)

        # 5. Display Result in a nice box
        st.divider()
        st.subheader("Results")
        st.metric(label="health status", value=f"{prediction[0]:.2f}")

        # Show how the input compares
        st.bar_chart(user_input.T)

except Exception as e:
    st.error(f"Model Error: {e}")
