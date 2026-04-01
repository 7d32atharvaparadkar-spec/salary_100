import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("salary_model.pkl","rb"))

st.title("💼 Salary Predictor (Advanced)")

exp = st.number_input("Experience (Years)", min_value=0.0)

edu = st.selectbox("Education", ["Graduate","Postgraduate","PhD"])

# Encoding manually
edu_post = 1 if edu == "Postgraduate" else 0
edu_phd = 1 if edu == "PhD" else 0

if st.button("Predict Salary"):
    
    data = np.array([[exp, edu_post, edu_phd]])
    
    result = model.predict(data)
    
    st.success(f"Estimated Salary = ₹ {result[0]:,.2f}")