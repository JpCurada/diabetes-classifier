import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.set_page_config(
  layout='centered',
  page_title='DEVCON Demo',
  )

model = load_model('diabetes_classifier')

st.title('What does your glucose tells you?')
st.caption('Diabetes mellitus remains a global health issue, causing several thousand people to die each day from this single condition. Finding and avoiding diabetes in the earlier stages can help reduce the risk of serious health issues such as circulatory system diseases, kidney malfunction, and vision loss. This project involves developing a predictive model for effectively detecting potential Diabetes cases, ideally, before commencing preventive treatment.')


# Collect user input
pregnancies = st.number_input('Total number of pregnancies', min_value=0.0)
glucose = st.number_input('Plasma glucose concentration after 2 hours in oral glucose tolerance test (mg/dl)', min_value=0.0)
blood_pressure = st.number_input('Diastolic blood pressure (mm Hg)', min_value=0.0)
skin_thickness = st.number_input('Triceps skinfold thickness (mm)', min_value=0.0)
insulin = st.number_input('2-Hour serum insulin	(μU/mL)', min_value=0.0)
bmi = st.number_input('Body mass index (kg/m²)', min_value=0.0)
diabetes_pedigree = st.number_input('Likelihood of diabetes based on family history', min_value=0.0)
age = st.number_input('Patient age', min_value=0.0)


# Predict the output
if st.button('Classify'):
  input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]],
  columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age',])
  prediction = predict_model(model, data=input_data)
  st.subheader(f"The Predicted Output is: {'Diabetic' if prediction['prediction_label'].iloc[0] == 1 else 'Non-Diabetic'}")
  st.write(f"Confidence Level: {round(prediction['prediction_score'].iloc[0] * 100, 2)}%")
  