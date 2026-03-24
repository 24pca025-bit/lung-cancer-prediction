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

        .banner-box {{
            background: rgba(255, 255, 255, 0.78);
            padding: 16px;
            border-radius: 18px;
            text-align: center;
            margin-bottom: 24px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.10);
            border: 1px solid rgba(16, 42, 67, 0.10);
        }}

        .intro-box {{
            background: rgba(255, 255, 255, 0.82);
            padding: 18px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            color: #243b53;
            font-size: 17px;
            line-height: 1.7;
        }}

        .section-title {{
            font-size: 28px;
            font-weight: 800;
            color: #1f3b73;
            margin-top: 20px;
            margin-bottom: 12px;
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
if "show_form" not in st.session_state:
    st.session_state.show_form = False

# ---------- Index / Home Section ----------
st.markdown(
    '<div class="banner-box">🫁 <b>AI-Based Lung Cancer Screening System</b> 🫁</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="main-title">Lung Cancer Prediction Using Machine Learning</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-text">Early screening support using machine learning and symptom-based prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="intro-box">
    <b>Welcome!</b><br><br>
    This application is developed to predict lung cancer based on patient symptoms and basic details.
    It uses a machine learning model trained on symptom data to classify whether lung cancer is present or not.
    <br><br>
    Please click the button below to enter patient details and continue to prediction.
    </div>
    """,
    unsafe_allow_html=True
)

if st.button("Start Prediction"):
    st.session_state.show_form = True

# ---------- Prediction Form Section ----------
if st.session_state.show_form:
    st.markdown('<div class="section-title">Patient Details and Symptoms</div>', unsafe_allow_html=True)

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

    if st.button("Predict"):
        input_data = np.array([[gender, age, smoking, fatigue, shortness_of_breath,
                                yellow_fingers, anxiety, peer_pressure,
                                chronic_disease, allergy, wheezing,
                                alcohol_consuming, coughing,
                                swallowing_difficulty, chest_pain]])

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            if name:
                st.error(f"Prediction Result for {name}: Lung Cancer")
            else:
                st.error("Prediction Result: Lung Cancer")
        else:
            if name:
                st.success(f"Prediction Result for {name}: No Lung Cancer")
            else:
                st.success("Prediction Result: No Lung Cancer")
            
