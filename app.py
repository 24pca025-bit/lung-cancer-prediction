import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("lung_cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Lung Cancer Prediction Using Machine Learning")
st.write("Enter patient symptom values (1 = No, 2 = Yes)")

# Input fields
yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])
anxiety = st.selectbox("Anxiety", [1, 2])
peer_pressure = st.selectbox("Peer Pressure", [1, 2])
chronic_disease = st.selectbox("Chronic Disease", [1, 2])
allergy = st.selectbox("Allergy", [1, 2])
wheezing = st.selectbox("Wheezing", [1, 2])
alcohol_consuming = st.selectbox("Alcohol Consuming", [1, 2])
coughing = st.selectbox("Coughing", [1, 2])
swallowing_difficulty = st.selectbox("Swallowing Difficulty", [1, 2])
chest_pain = st.selectbox("Chest Pain", [1, 2])

# Convert UI values: 1 = No, 2 = Yes  -->  0 = No, 1 = Yes
def convert_input(value):
    return 0 if value == 1 else 1

if st.button("Predict"):
    input_data = np.array([[
        convert_input(yellow_fingers),
        convert_input(anxiety),
        convert_input(peer_pressure),
        convert_input(chronic_disease),
        convert_input(allergy),
        convert_input(wheezing),
        convert_input(alcohol_consuming),
        convert_input(coughing),
        convert_input(swallowing_difficulty),
        convert_input(chest_pain)
    ]])

       prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Prediction Result: High Risk of Lung Cancer")
    else:
        st.success("Prediction Result: Low Risk of Lung Cancer")
