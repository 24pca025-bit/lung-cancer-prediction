import streamlit as st
import pickle
import numpy as np

# Custom UI styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to bottom, #f7fbff, #eef6ff);
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 800;
        color: #1f3b73;
        margin-bottom: 10px;
    }

    .sub-text {
        text-align: center;
        font-size: 18px;
        color: #4a5a6a;
        margin-bottom: 30px;
    }

    .banner-box {
        background-color: #dff1ff;
        padding: 18px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }

    div.stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 10px;
        padding: 10px 22px;
        border: none;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #145a86;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved model
with open("lung_cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# Banner and title
st.markdown('<div class="banner-box">🫁 <b>Lung Health Screening App</b> | Machine Learning Based Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">Lung Cancer Prediction Using Machine Learning</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Enter patient symptom values and click Predict to view the result.</div>', unsafe_allow_html=True)

# Input fields
gender = st.selectbox("Gender (Male=1, Female=0)", [1, 0])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
smoking = st.selectbox("Smoking (1 = No, 2 = Yes)", [1, 2])
fatigue = st.selectbox("Fatigue (1 = No, 2 = Yes)", [1, 2])
shortness_of_breath = st.selectbox("Shortness of Breath (1 = No, 2 = Yes)", [1, 2])

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

# Prediction
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
