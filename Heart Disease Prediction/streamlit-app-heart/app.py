import os
import joblib
import pandas as pd
import streamlit as st

model = joblib.load(os.path.join(os.path.dirname(__file__), "../svc_trained_model.pkl"))

st.title("Heart Disease Prediction System")

age = st.number_input("Age", 20, 100, 50)
cp = st.selectbox("Chest Pain Type", [1, 2, 3, 4],
                  format_func=lambda x: {1: "Typical angina", 2: "Atypical angina",
                                         3: "Non-anginal pain", 4: "Asymptomatic"}[x])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)

if st.button("Predict"):
    X = pd.DataFrame([[age, cp, thalach, oldpeak]],
                     columns=["age", "cp", "thalach", "oldpeak"])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    if pred == 1:
        st.error(f"Heart disease likely (probability {prob:.2f})")
    else:
        st.success(f"Heart disease unlikely (probability {prob:.2f})")