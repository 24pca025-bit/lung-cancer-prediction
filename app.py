import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("lung_cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Lung Cancer Prediction Using Machine Learning")
st.write("Enter patient symptom values")

# New 5 features added
gender = st.selectbox("Gender (Male=1, Female=0)", [1, 0])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
smoking = st.selectbox("Smoking (1 = No, 2 = Yes)", [1, 2])
fatigue = st.selectbox("Fatigue (1 = No, 2 = Yes)", [1, 2])
shortness_of_breath = st.selectbox("Shortness of Breath (1 = No, 2 = Yes)", [1, 2])

# Existing 10 features
yellow_fingers = st.selectbox("Yellow Fingers (1 = No, 2 = Yes)", [1, 2])
anxiety = st.selectbox("Anxiety (1 = No, 2 = Yes)", [1, 2])
peer_pressure = st.selectbox("Peer Pressure (1 = No, 2 = Yes)", [1, 2])
chronic_disease = st.selectbox("Chronic Disease (1 = No, 2 = Yes)", [1, 2])
allergy = st.selectbox("Allergy (1 = No, 2 = Yes)", [1, 2])
wheezing = st.selectbox("Wheezing (1 = No, 2 = Yes)", [1, 2])
alcohol_consuming = st.selectbox("Alcohol Consuming (1 = No, 2 = Yes)", [1, 2])
coughing = st.selectbox("Coughing (1 = No, 2 = Yes)", [1, 2])
swallowing_difficulty = st.selectbox("Swallowing Difficulty (1 = No, 2 = Yes)", [1, 2])
chest_pain = st.selectbox("Chest Pain (1 = No, 2 = Yes)", [1, 2])

if st.button("Predict"):
    input_data = np.array([[gender, age, smoking, fatigue, shortness_of_breath,
                            yellow_fingers, anxiety, peer_pressure,
                            chronic_disease, allergy, wheezing,
                            alcohol_consuming, coughing,
                            swallowing_difficulty, chest_pain]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction Result: Yes - Lung Cancer Detected")
    else:
        st.success("Prediction Result: No - Lung Cancer Not Detected")
    
