import streamlit as st
import pandas as pd
import pickle

# -------------------------
# Load model and encoders
# -------------------------
@st.cache_resource
def load_model():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], model_data["features_names"], encoders

model, feature_names, encoders = load_model()

# -------------------------
# Retention strategy function
# -------------------------
def suggest_retention_strategies(customer_row):
    suggestions = []

    if customer_row['MonthlyCharges'] > 150:
        suggestions.append("Offer discount or bundle services to reduce monthly cost.")

    if customer_row['Contract'] == "Month-to-month":
        suggestions.append("Promote longer-term contracts with discounts.")

    if customer_row['TechSupport'] == "No":
        suggestions.append("Offer free or discounted Tech Support trial.")

    if customer_row['OnlineSecurity'] == "No":
        suggestions.append("Provide free Online Security for 3 months.")

    if customer_row['SeniorCitizen'] == 1 and customer_row['tenure'] < 12:
        suggestions.append("Assign a dedicated support representative.")

    if not suggestions:
        suggestions.append("Maintain engagement with regular offers and communication.")

    return suggestions

# -------------------------
# Streamlit UI
# -------------------------
st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details below to predict churn and suggest retention strategies.")

# Input form
with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Prepare input
    input_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical variables
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    # Prediction
    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1] * 100

    if prediction[0] == 1:
        st.error(f"🚨 Customer is **likely to churn** (Probability: {prob:.2f}%)")
        st.subheader("🔎 Suggested Retention Strategies")
        strategies = suggest_retention_strategies(input_data)
        for s in strategies:
            st.write("-", s)
    else:
        st.success(f"✅ Customer is **unlikely to churn** (Probability: {prob:.2f}%)")
        st.write("Keep up the good service!")
