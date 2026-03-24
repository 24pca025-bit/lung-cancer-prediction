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
            width: 100%;
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

    st.mark
