import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load model
# ----------------------------
model = joblib.load(r"model\rf_binary_model.pkl")

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Student Pass/Fail Predictor",
    page_icon="🎓",
    layout="centered"
)

# ----------------------------
# Title
# ----------------------------
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🎓 Student Pass/Fail Predictor</h1>", unsafe_allow_html=True)
st.write("---")

# ----------------------------
# Input form in two columns
# ----------------------------
st.header("Enter Student Details:")

col1, col2 = st.columns(2)

with col1:
    study_hours = st.slider("📚 Study Hours per Week", 0, 100, 10)
    attendance = st.slider("🕒 Attendance Rate (%)", 0, 100, 75)
    previous_grades = st.slider("📝 Previous Grades (%)", 0, 100, 60)

with col2:
    participation = st.selectbox("🎭 Extracurricular Participation", ["No", "Yes"])
    parent_edu = st.selectbox("👨‍🎓 Parent Education Level", ["High School", "Bachelor's", "Master's", "Other"])

# ----------------------------
# Derived features
# ----------------------------
study_efficiency = previous_grades / (study_hours + 1)
academic_engagement = (attendance * study_hours) / 100

def categorize_study_hours(hours):
    if hours < 10:
        return 'Low'
    elif hours < 20:
        return 'Medium'
    else:
        return 'High'

def categorize_attendance(rate):
    if rate < 70:
        return 'Poor'
    elif rate < 85:
        return 'Average'
    else:
        return 'Good'

study_cat = categorize_study_hours(study_hours)
att_cat = categorize_attendance(attendance)

# ----------------------------
# Input DataFrame
# ----------------------------
input_df = pd.DataFrame({
    'Study Hours per Week': [study_hours],
    'Attendance Rate': [attendance],
    'Previous Grades': [previous_grades],
    'Study Efficiency': [study_efficiency],
    'Academic Engagement': [academic_engagement],
    'Participation in Extracurricular Activities': [participation],
    'Parent Education Level': [parent_edu],
    'Study Hours Category': [study_cat],
    'Attendance Category': [att_cat]
})

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict 🚀"):
    pred = model.predict(input_df)[0]
    proba_array = model.predict_proba(input_df)[0]

# Get probability of the predicted class
    pred_prob = proba_array[list(model.classes_).index(pred)]


    st.write("---")
    if pred == 1:
        st.success(f"✅ Prediction: Pass")
        st.progress(int(pred_prob * 100))
        st.info(f"Confidence: {pred_prob:.2f}")
    else:
        st.error(f"❌ Prediction: Fail")
        st.progress(int(pred_prob * 100))
        st.info(f"Confidence: {pred_prob:.2f}")

# ----------------------------
# Footer
# ----------------------------
st.write("---")
