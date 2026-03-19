
import streamlit as st
import pickle
import numpy as np

# Load saved model
with open("lung_cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Lung Cancer Prediction Using Machine Learning")
st.write("Enter patient symptom values (1 = No, 2 = Yes)")

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

if st.button("Predict"):
    input_data = np.array([[yellow_fingers, anxiety, peer_pressure,
                            chronic_disease, allergy, wheezing,
                            alcohol_consuming, coughing,
                            swallowing_difficulty, chest_pain]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("Prediction Result: Lung Cancer Detected")
    else:
        st.success("Prediction Result: No Lung Cancer")
