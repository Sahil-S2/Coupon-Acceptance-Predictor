# 📊 Coupon Acceptance Predictor with SHAP
import streamlit as st
import shap
import matplotlib.pyplot as plt
from utils import load_model, make_prediction, explain_prediction
from streamlit_lottie import st_lottie
import requests
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*ScriptRunContext.*")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def main():
    # Page config
    st.set_page_config(page_title="🎯 Coupon Acceptance Predictor", layout="centered")

    # Load animated Lottie logo
    lottie_logo = load_lottieurl("https://assets9.lottiefiles.com/private_files/lf30_oqpbtola.json")

    # Centered animation
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st_lottie(lottie_logo, height=150, key="coupon")

    # Styled title and subtitle
    st.markdown(
        """
        <h1 style='text-align: center; color: #3E64FF;'>🎯 Coupon Acceptance Predictor</h1>
        <p style='text-align: center; color: gray;'>AI-powered prediction based on customer behavior & profile</p>
        """,
        unsafe_allow_html=True
    )

    # ------------------------- Load Assets -------------------------
    model, preprocessor, threshold = load_model()

    # ------------------------- Sidebar Inputs -------------------------
    st.sidebar.header("📝 Customer Profile")

    user_input = {
        'destination': st.sidebar.selectbox("Destination", ['No Urgent Place', 'Home', 'Work']),
        'passanger': st.sidebar.selectbox("Passenger", ['Alone', 'Friend(s)', 'Partner']),
        'weather': st.sidebar.selectbox("Weather", ['Sunny', 'Rainy', 'Snowy']),
        'temperature': st.sidebar.slider("Temperature (°F)", 30, 100, 65),
        'coupon': st.sidebar.selectbox("Coupon Type", ['Coffee House', 'Restaurant(<20)', 'Carry out & Take away']),
        'expiration': st.sidebar.selectbox("Coupon Expiration", ['1d', '2h']),
        'gender': st.sidebar.selectbox("Gender", ['Male', 'Female']),
        'age': st.sidebar.slider("Age", 16, 75, 30),
        'maritalStatus': st.sidebar.selectbox("Marital Status", ['Single', 'Unmarried partner', 'Married partner', 'Divorced']),
        'has_children': st.sidebar.radio("Has Children?", [0, 1]),
        'education': st.sidebar.selectbox("Education", [
            'Some college - no degree', 'Bachelors degree', 'Associates degree',
            'High School Graduate', 'Graduate degree (Masters or Doctorate)', 'Some High School'
        ]),
        'occupation': st.sidebar.selectbox("Occupation", ['Unemployed', 'Student', 'Professional', 'Sales', 'Other']),
        'income': st.sidebar.selectbox("Income Range", [
            'Less than $12500', '$12500 - $24999', '$25000 - $37499', '$37500 - $49999'
        ]),
        'car': st.sidebar.selectbox("Car Ownership", ['None', '1 car', '2 cars']),
        'Bar': st.sidebar.selectbox("Bar Visit Frequency", ['never', 'less1', '1~3', '4~8', 'gt8']),
        'CoffeeHouse': st.sidebar.selectbox("CoffeeHouse Visit Frequency", ['never', 'less1', '1~3', '4~8', 'gt8']),
        'CarryAway': st.sidebar.selectbox("Takeaway Frequency", ['never', 'less1', '1~3', '4~8', 'gt8']),
        'RestaurantLessThan20': st.sidebar.selectbox("Restaurant (<$20) Frequency", ['never', 'less1', '1~3', '4~8', 'gt8']),
        'Restaurant20To50': st.sidebar.selectbox("Restaurant ($20-$50) Frequency", ['never', 'less1', '1~3', '4~8', 'gt8']),
        'toCoupon_GEQ5min': st.sidebar.radio("Time to Coupon ≥ 5 min?", [0, 1]),
        'toCoupon_GEQ15min': st.sidebar.radio("Time to Coupon ≥ 15 min?", [0, 1]),
        'toCoupon_GEQ25min': st.sidebar.radio("Time to Coupon ≥ 25 min?", [0, 1]),
    }

    # ------------------------- Predict & Explain -------------------------
    if st.button("🚀 Predict"):
        pred, prob = make_prediction(model, preprocessor, threshold, user_input)

        st.markdown("## 🔍 Prediction Result")
        if pred:
            st.success("✅ The user is **likely to ACCEPT** the coupon.")
        else:
            st.error("❌ The user is **likely to REJECT** the coupon.")
        st.markdown(f"**Probability of Acceptance:** `{prob:.2f}`")

        # --------------------- SHAP Explanation ---------------------
        st.markdown("## 📈 Feature Contribution (SHAP)")
        shap_values, processed_input = explain_prediction(model, preprocessor, user_input)

        fig, ax = plt.subplots(figsize=(10, 1.5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

    # ------------------------- Footer -------------------------
    st.markdown("---")
    st.caption("🧠 Built as part of a Data Science placement project. | ✨ Developed using Streamlit + SHAP")

# Needed to run on Hugging Face Spaces or any server environment
if __name__ == "__main__":
    main()







