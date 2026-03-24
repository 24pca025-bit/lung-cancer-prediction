import streamlit as st
import pickle
import numpy as np
import base64

# ---------- Page Config ----------
st.set_page_config(page_title="Lung Cancer Prediction", page_icon="🫁", layout="centered")

# ---------- Background Image Function ----------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255,255,255,0.82), rgba(255,255,255,0.88)),
                              url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .main-title {{
            text-align: center;
            font-size: 52px;
            font-weight: 900;
            color: #102a43;
            margin-top: 10px;
            margin-bottom: 8px;
            text-shadow: 1px 1px 2px rgba(255,255,255,0.7);
        }}

        .sub-text {{
            text-align: center;
            font-size: 18px;
            color: #243b53;
            margin-bottom: 28px;
            font-weight: 500;
        }}

        .section-title {{
            font-size: 28px;
            font-weight: 800;
            color: #1f3b73;
            margin-top: 20px;
            margin-bottom: 12px;
            text-align: center;
        }}

        .result-box {{
            background: rgba(255,255,255,0.85);
            padding: 22px;
            border-radius: 18px;
            margin-top: 20px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
            text-align: center;
        }}

        div.stButton > button {{
            background: linear-gradient(to right, #0f4c81, #1f77b4);
            color: white;
            border-radius: 12px;
            padding: 10px 24px;
            border: none;
            font-weight: 700;
            font-size: 16px;
        }}

        div.stButton > button:hover {{
            background: linear-gradient(to right, #0b3c66, #145a86);
            color: white;
        }}

        [data-baseweb="select"] > div,
        .stNumberInput > div > div > input,
        .stTextInput > div > div > input {{
            background-color: rgba(255,255,255,0.88);
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ---------- Apply Background ----------
add_bg_from_local("lungs_bg.jpg")

# ---------- Load Model ----------
with open("lung_cancer_rf_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------- Session State ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "patient_name" not in st.session_state:
    st.session_state.patient_name = ""

# ---------- HOME PAGE ----------
if st.session_state.page == "home":
    
    st.markdown(
        '<div class="main-title">Lung Cancer Prediction Using Machine Learning</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-text">Early screening support using machine learning and symptom-based prediction</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Start Prediction"):
            st.session_state.page = "prediction"
            st.rerun()

# ---------- PREDICTION PAGE ----------
elif st.session_state.page == "prediction":
   
    st.markdown(
        '<div class="section-title">Enter Patient Details and Symptoms</div>',
        unsafe_allow_html=True
    )

    name = st.text_input("Patient Name")

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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict"):
            input_data = np.array([[gender, age, smoking, fatigue, shortness_of_breath,
                                    yellow_fingers, anxiety, peer_pressure,
                                    chronic_disease, allergy, wheezing,
                                    alcohol_consuming, coughing,
                                    swallowing_difficulty, chest_pain]])

            prediction = model.predict(input_data)

            st.session_state.patient_name = name

            if prediction[0] == 1:
                st.session_state.prediction_result = "Lung Cancer"
            else:
                st.session_state.prediction_result = "No Lung Cancer"

            st.session_state.page = "result"
            st.rerun()

    with col2:
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()

# ---------- RESULT PAGE ----------
elif st.session_state.page == "result":
    
    st.markdown(
        '<div class="section-title">Final Result</div>',
        unsafe_allow_html=True
    )

    patient_name = st.session_state.patient_name
    result = st.session_state.prediction_result

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    if patient_name:
        st.write(f"### Patient Name: {patient_name}")

    if result == "Lung Cancer":
        st.error("Prediction Result: Lung Cancer")
    else:
        st.success("Prediction Result: No Lung Cancer")

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Prediction Page"):
            st.session_state.page = "prediction"
            st.rerun()

    with col2:
        if st.button("Go to Home Page"):
            st.session_state.page = "home"
            st.rerun()
