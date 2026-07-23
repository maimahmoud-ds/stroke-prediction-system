import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================
# Load Model Files
# ==========================

model = joblib.load("stroke_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")


# ==========================
# Page Config
# ==========================

st.set_page_config(
    page_title="Stroke Prediction AI",
    page_icon="🧠",
    layout="centered"
)


# ==========================
# Custom CSS
# ==========================

st.markdown(
"""
<style>

.stApp {
    background-color: #ffffff;
}

h1 {
    color: #0052cc;
    font-weight: 800;
}

.subtitle {
    color: #4c6b8a;
    font-size: 16px;
    font-weight: 600;
}

.card-box {
    background:white;
    padding:22px;
    border-radius:18px;
    border:1px solid #e1e9f5;
    box-shadow:0px 8px 20px rgba(0,82,204,0.06);
    margin-bottom:25px;
}

.result-card-high {
    background:#fff8f8;
    padding:24px;
    border-radius:18px;
    border:2px solid #ff4d4f;
}

.result-card-low {
    background:#f6ffed;
    padding:24px;
    border-radius:18px;
    border:2px solid #52c41a;
}

.stButton button {
    width:100%;
    height:55px;
    border-radius:14px;
    background:#0066ff;
    color:white;
    font-size:19px;
    font-weight:700;
}

.stButton button:hover {
    background:#0052cc;
}

</style>
""",
unsafe_allow_html=True
)


# ==========================
# Header
# ==========================

col1, col2 = st.columns([1, 3.5], vertical_alignment="center")

with col1:
    st.image(
        "logo.png",
        use_container_width=True
    )

with col2:
    st.title("Stroke Prediction AI")
    st.markdown(
        '<p class="subtitle">⚡ Real-time Smart Health Assessment</p>',
        unsafe_allow_html=True
    )

st.write("")


# ==========================
# Patient Information
# ==========================

st.markdown(
"""
<div class="card-box">
<h3 style="color:#0052cc; margin-top:0;">
🩺 Patient Information
</h3>
</div>
""",
unsafe_allow_html=True
)

col_a, col_b = st.columns(2)

with col_a:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    gender = st.selectbox("Gender", ["Female", "Male"])
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    ever_married = st.selectbox("Ever Married", ["No", "Yes"])

with col_b:
    work_type = st.selectbox("Work Type", ["Private", "Self-employed", "children", "Never_worked", "Govt_job"])
    residence = st.selectbox("Residence Type", ["Urban", "Rural"])
    glucose = st.number_input("Average Glucose Level", min_value=0.0, value=100.0)
    bmi = st.number_input("BMI", min_value=0.0, value=25.0)
    smoking = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])


# ==========================
# Prediction
# ==========================

if st.button("🔍 Predict Stroke Risk"):

    # 1️⃣ إنشاء الـ DataFrame للمريض
    data = pd.DataFrame({
        "age": [age],
        "gender": [gender],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "ever_married": [ever_married],
        "work_type": [work_type],
        "Residence_type": [residence],
        "avg_glucose_level": [glucose],
        "bmi": [bmi],
        "smoking_status": [smoking]
    })

    # 2️⃣ Feature Engineering
    data["risk_combo"] = data["hypertension"] + data["heart_disease"]
    data["avg_glucose_level"] = np.log1p(data["avg_glucose_level"])

    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 40, 60, 120],
        labels=["young", "middle", "senior"]
    )

    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["underweight", "normal", "overweight", "obese"]
    )

    data["bmi_category"] = (
        data["bmi_category"]
        .cat.add_categories("Missing")
        .fillna("Missing")
    )

    # 3️⃣ Encoding
    data = pd.get_dummies(
        data,
        columns=[
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status",
            "age_group",
            "bmi_category"
        ],
        drop_first=True
    )

    # 💡 معالجة عمود Unnamed: 1 لو كان موجود في features.pkl
    if "Unnamed: 1" in features and "Unnamed: 1" not in data.columns:
        data["Unnamed: 1"] = 0

    # 4️⃣ المطابقة مع أعمدة التدريب وتحويل النوع
    data = data.reindex(columns=features, fill_value=0)
    data = data.astype(float)

    
    

    # 5️⃣ Scaling
    data_scaled = scaler.transform(data)

    # 6️⃣ حساب الاحتمالية والتنبؤ
    probability = model.predict_proba(data_scaled)[0][1]
    threshold = 0.30 
    prediction = 1 if probability >= threshold else 0
    risk = probability * 100

    st.divider()

    # ==========================
    # Results
    # ==========================
    if prediction == 1:
        st.markdown(
            f"""
            <div class="result-card-high">
                <h2 style="color:#e53935; margin-top:0;">⚠️ High Risk of Stroke</h2>
                <h3>Risk Probability: {risk:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🩺 Medical Recommendations")
        tips = [
            "Consult a healthcare professional.",
            "Monitor blood pressure regularly.",
            "Control blood glucose levels.",
            "Reduce salt and unhealthy fats.",
            "Exercise regularly.",
            "Avoid smoking."
        ]
        for tip in tips:
            st.warning("✔️ " + tip)

    else:
        st.markdown(
            f"""
            <div class="result-card-low">
                <h2 style="color:#2e7d32; margin-top:0;">✅ Low Risk of Stroke</h2>
                <h3>Risk Probability: {risk:.2f}%</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("💙 Healthy Lifestyle Tips")
        tips = [
            "Maintain a balanced diet.",
            "Exercise at least 30 minutes daily.",
            "Keep blood pressure normal.",
            "Do regular health checkups.",
            "Drink enough water.",
            "Maintain good sleep habits."
        ]
        for tip in tips:
            st.success("✔️ " + tip)
