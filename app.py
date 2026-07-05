import tensorflow as tf
import keras

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

# ---------------- Page Configuration ---------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------- Load Model ---------------- #
model = tf.keras.models.load_model("model.h5")

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------- Title ---------------- #
st.title("📊 Customer Churn Prediction System")
st.markdown(
    "Predict whether a bank customer is likely to **leave the bank** based on their profile."
)

st.divider()

# ---------------- Input Section ---------------- #
col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92, 30)
    credit_score = st.number_input("💳 Credit Score", value=600)

with col2:
    balance = st.number_input("🏦 Balance", value=0.0)
    estimated_salary = st.number_input("💰 Estimated Salary", value=50000.0)
    tenure = st.slider("📅 Tenure", 0, 10, 5)
    num_of_products = st.slider("📦 Number of Products", 1, 4, 1)

has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
is_active_member = st.selectbox("✅ Is Active Member", [0, 1])

st.divider()

# ---------------- Prediction ---------------- #
if st.button("🔍 Predict Churn", use_container_width=True):

    input_data = pd.DataFrame({
        "CreditScore": [credit_score],
        "Gender": [label_encoder_gender.transform([gender])[0]],
        "Geography": [geography],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_cr_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    geo_encoded = onehot_encoder_geo.transform(
        input_data[["Geography"]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
    )

    input_data = input_data.drop("Geography", axis=1)

    input_data = pd.concat(
        [input_data.reset_index(drop=True), geo_encoded_df],
        axis=1
    )

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)

    prediction_proba = prediction[0][0]

    st.subheader("Prediction Result")

    st.progress(float(prediction_proba))

    st.metric(
        "Churn Probability",
        f"{prediction_proba*100:.2f}%"
    )

    if prediction_proba > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")