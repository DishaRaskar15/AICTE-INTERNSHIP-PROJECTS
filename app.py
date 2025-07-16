import streamlit as st
import numpy as np
import joblib
import os

# ---------- Page Config ----------
st.set_page_config(page_title="Enhanced Salary Predictor", layout="centered")
st.title("💼 Smart Salary Predictor")
st.markdown("Please fill in your details to get a personalized salary prediction.")

# ---------- Load Model ----------
model = None
model_path = "salary_model.pkl"

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.warning("⚠️ Model file not found. Using default model for testing.")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    X_dummy = np.array([[1, 1, 25, 1], [2, 2, 30, 2], [0, 0, 22, 0]])
    y_dummy = [300000, 600000, 200000]
    model.fit(X_dummy, y_dummy)

# ---------- Encoding Maps ----------
education_map = {
    "High School": 0,
    "Bachelors": 1,
    "Masters": 2,
    "PhD": 3
}

branch_map = {
    "IT": 0,
    "CSE": 1,
    "ECE": 2,
    "EEE": 3,
    "MECH": 4,
    "CIVIL": 5,
    "OTHER": 6
}

# ---------- Collect User Info ----------
with st.form("salary_form"):
    name = st.text_input("Your Name")
    age = st.number_input("Age", min_value=18, max_value=65, step=1)
    branch = st.selectbox("Branch of Study", list(branch_map.keys()))
    experience = st.number_input("Years of Experience", min_value=0.0, step=0.5)
    education = st.selectbox("Education Level", list(education_map.keys()))

    submitted = st.form_submit_button("Predict Salary 💰")

if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# ---------- Save inputs after form submission ----------
if submitted:
    st.session_state.submitted = True
    st.session_state.name = name
    st.session_state.age = age
    st.session_state.branch = branch
    st.session_state.experience = experience
    st.session_state.education = education

# ---------- Prediction & Flash Card ----------
if st.session_state.submitted:
    try:
        branch_encoded = branch_map[st.session_state.branch]
        education_encoded = education_map[st.session_state.education]
        input_features = np.array([[st.session_state.experience, education_encoded, st.session_state.age, branch_encoded]])
        predicted_salary = model.predict(input_features)[0]

        # ---------- Flash Card Display ----------
        with st.container():
            st.markdown("---")
            st.subheader("📋 Salary Prediction Flash Card")
            st.markdown(f"""
                <div style="border: 2px solid #4CAF50; padding: 20px; border-radius: 15px; background-color: #f9f9f9;">
                    <h3>👤 Name: <span style="color: #2e86de;">{st.session_state.name}</span></h3>
                    <p><strong>Age:</strong> {st.session_state.age} | <strong>Branch:</strong> {st.session_state.branch}</p>
                    <p><strong>Experience:</strong> {st.session_state.experience} years | <strong>Education:</strong> {st.session_state.education}</p>
                    <h2>💰 Estimated Salary: ₹{predicted_salary:,.2f}</h2>
                </div>
            """, unsafe_allow_html=True)

            st.info("💡 This estimate is based on your profile. Real-world salaries may vary.")

            if predicted_salary < 300000:
                st.warning("🚀 Consider enhancing your skills or switching roles.")
            elif predicted_salary < 800000:
                st.success("📈 You are on a great path. Keep improving!")
            else:
                st.balloons()
                st.success("🔥 You’re a top performer!")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
