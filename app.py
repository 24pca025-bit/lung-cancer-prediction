import streamlit as st
import pickle
import numpy as np
import base64
import os

# ---------- Page Config ----------
st.set_page_config(
    page_title="Lung Cancer Prediction",
    page_icon="🫁",
    layout="centered"
)

# ---------- Background Image ----------
def add_bg_from_local(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as image:
            encoded_string = base64.b64encode(image.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(0,0,0,0.45), rgba(0,0,0,0.45)),
                                  url("data:image/jpg;base64,{encoded_string}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}

            .main-title {{
                text-align: center;
                font-size: 52px;
                font-weight: 900;
                color: white;
                margin-top: 10px;
                margin-bottom: 8px;
                text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
            }}

            .sub-text {{
                text-align: center;
                font-size: 18px;
                color: #f1f5f9;
                margin-bottom: 25px;
                font-weight: 500;
                text-shadow: 1px 1px 4px rgba(0,0,0,0.5);
            }}

            .section-title {{
                font-size: 28px;
                font-weight: 800;
                color: white;
                margin-top: 20px;
                margin-bottom: 16px;
                text-align: center;
                text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
            }}

            .patient-name-box {{
                width: 100%;
                background: rgba(255,255,255,0.96);
                color: #102a43;
                font-size: 26px;
                font-weight: 800;
                padding: 20px 24px;
                border-radius: 24px;
                margin-top: 12px;
                margin-bottom: 20px;
                box-shadow: 0 8px 22px rgba(0,0,0,0.20);
                text-align: left;
            }}

            .positive-result {{
                width: 100%;
                background: rgba(255, 230, 230, 0.95);
                color: #b30000;
                font-size: 28px;
                font-weight: 800;
                padding: 22px;
                border-radius: 20px;
                margin-top: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                text-align: left;
            }}

            .negative-result {{
                width: 100%;
                background: rgba(232, 255, 240, 0.95);
                color: #0b7a33;
                font-size: 28px;
                font-weight: 800;
                padding: 22px;
                border-radius: 20px;
                margin-top: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.12);
                text-align: left;
            }}

            .note-text {{
                color: #0b2240;
                font-size: 18px;
                margin-top: 14px;
                font-weight: 700;
                text-align: left;
                background: rgba(255,255,255,0.20);
                padding: 10px 12px;
                border-radius: 10px;
            }}

            label, .stTextInput label, .stNumberInput label, .stSelectbox label {{
                color: white !important;
                font-weight: 600 !important;
                text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
            }}

            .stTextInput div[data-testid="stWidgetLabel"] label,
            .stNumberInput div[data-testid="stWidgetLabel"] label,
            .stSelectbox div[data-testid="stWidgetLabel"] label {{
                color: white !important;
                font-weight: 600 !important;
                text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
            }}

            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            [data-baseweb="select"] > div {{
                background-color: rgba(255,255,255,0.95);
                border-radius: 10px;
            }}

            div.stButton > button {{
                border-radius: 14px;
                font-weight: 700;
                border: none;
                width: 100%;
                min-height: 52px;
                font-size: 18px;
                background: rgba(255,255,255,0.95);
                color: #1f2d3d;
            }}

            div.stButton > button:hover {{
                color: white !important;
                background: linear-gradient(to right, #0b3c66, #145a86) !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Background image file '{image_file}' not found.")

# ---------- Apply Background ----------
add_bg_from_local("lungs_bg.jpg")

# ---------- Cached Model Load ----------
@st.cache_resource
def load_model():
    model_file = "lung_cancer_rf_model.pkl"
    if not os.path.exists(model_file):
        return None, f"Model file '{model_file}' not found."
    try:
        with open(model_file, "rb") as file:
            model = pickle.load(file)
        return model, None
    except Exception as e:
        return None, f"Model loading error: {e}"

model, model_error = load_model()

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
        '<div class="main-title">LUNG CANCER PREDICTION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-text">Early screening support using machine learning and symptom-based prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:120px;'></div>", unsafe_allow_html=True)

    left_col, center_col, right_col = st.columns([2.5, 2, 2.5])

    with center_col:
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] > button {
                min-height: 60px;
                font-size: 18px;
                background: linear-gradient(to right, #0f4c81, #1f77b4);
                color: white;
                box-shadow: 0 6px 16px rgba(0,0,0,0.25);
            }
            div[data-testid="stButton"] > button:hover {
                background: linear-gradient(to right, #0b3c66, #145a86);
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        if st.button("Start Prediction", key="start_prediction_btn"):
            st.session_state.page = "prediction"

# ---------- PREDICTION PAGE ----------
elif st.session_state.page == "prediction":

    st.markdown(
        '<div class="section-title">Enter Patient Details and Symptoms</div>',
        unsafe_allow_html=True
    )

    if model_error:
        st.error(model_error)

    name = st.text_input("Patient Name")

    gender = st.selectbox("Gender (Male = 1, Female = 0)", [1, 0])
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

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Predict", key="predict_btn"):
            if model is None:
                st.error("Model not loaded properly.")
            elif not name.strip():
                st.warning("Please enter patient name.")
            else:
                input_data = np.array([[
                    gender,
                    age,
                    smoking,
                    fatigue,
                    shortness_of_breath,
                    yellow_fingers,
                    anxiety,
                    peer_pressure,
                    chronic_disease,
                    allergy,
                    wheezing,
                    alcohol_consuming,
                    coughing,
                    swallowing_difficulty,
                    chest_pain
                ]])

                try:
                    prediction = model.predict(input_data)
                    st.session_state.patient_name = name.strip()

                    if prediction[0] == 1:
                        st.session_state.prediction_result = "Lung Cancer"
                    else:
                        st.session_state.prediction_result = "No Lung Cancer"

                    st.session_state.page = "result"
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    with col2:
        if st.button("Back to Home", key="back_home_btn"):
            st.session_state.page = "home"
            st.rerun()

# ---------- RESULT PAGE ----------
elif st.session_state.page == "result":

    patient_name = st.session_state.patient_name
    result = st.session_state.prediction_result

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    # Patient name ONLY inside white box
    st.markdown(
        f"""
        <div class="patient-name-box">
            Patient Name: {patient_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    if result == "Lung Cancer":
        st.markdown(
            '<div class="positive-result">Prediction Result: Lung Cancer</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="note-text">Please consult a doctor for further medical evaluation.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="negative-result">Prediction Result: No Lung Cancer</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<div class="note-text">Currently, no lung cancer risk is predicted based on the given symptoms.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Prediction Page", key="back_prediction_btn"):
            st.session_state.page = "prediction"
            st.rerun()

    with col2:
        if st.button("Go to Home Page", key="go_home_btn"):
            st.session_state.page = "home"
            st.rerun()
