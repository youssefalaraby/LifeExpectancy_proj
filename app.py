import streamlit as st
import joblib
import pandas as pd

model = joblib.load("voting_regressor_model.pkl")

def predict_life_expectancy(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return prediction[0]

st.title("Life Expectancy Prediction App")

st.header("Enter the details below:")

year = st.number_input("Year", min_value=2000, max_value=2024, step=1, value=2024)
status=st.selectbox('Status',{'Developing','Developed'})
adult_mortality = st.number_input("Adult Mortality", min_value=0.0, value=100.0)
infant_deaths = st.number_input("Infant Deaths", min_value=0.0, value=0.0)
alcohol = st.number_input("Alcohol", min_value=0.0, value=1.0)
percentage_expenditure = st.number_input("Percentage Expenditure", min_value=0.0, value=100.0)
hepatitis_b = st.number_input("Hepatitis B", min_value=0.0, max_value=100.0, value=90.0)
measles = st.number_input("Measles", min_value=0.0, value=0.0)
bmi = st.number_input("BMI", min_value=0.0, value=20.0)
under_five_deaths = st.number_input("Under-five Deaths", min_value=0.0, value=0.0)
polio = st.number_input("Polio", min_value=0.0, max_value=100.0, value=90.0)
total_expenditure = st.number_input("Total Expenditure", min_value=0.0, value=6.0)
diphtheria = st.number_input("Diphtheria", min_value=0.0, max_value=100.0, value=90.0)
hiv_aids = st.number_input("HIV/AIDS", min_value=0.0, value=0.1)
gdp = st.number_input("GDP", min_value=0.0, value=1000.0)
population = st.number_input("Population", min_value=0.0, value=1000000.0)
thinness_1_19_years = st.number_input("Thinness 1-19 years", min_value=0.0, value=5.0)
thinness_5_9_years = st.number_input("Thinness 5-9 years", min_value=0.0, value=5.0)
income_composition_of_resources = st.number_input("Income Composition of Resources", min_value=0.0, max_value=1.0, value=0.7)
schooling = st.number_input("Schooling", min_value=0.0, value=12.0)


if st.button("Predict Life Expectancy"):
    # Collecting the input data into a dictionary
    input_data = {
        "Year": year,
        "Status":status,
        "Adult Mortality": adult_mortality,
        "Infant Deaths": infant_deaths,
        "Alcohol": alcohol,
        "Percentage Expenditure": percentage_expenditure,
        "Hepatitis B": hepatitis_b,
        "Measles": measles,
        "BMI": bmi,
        "Under-five Deaths": under_five_deaths,
        "Polio": polio,
        "Total Expenditure": total_expenditure,
        "Diphtheria": diphtheria,
        "HIV/AIDS": hiv_aids,
        "GDP": gdp,
        "Population": population,
        "Thinness 1-19 years": thinness_1_19_years,
        "Thinness 5-9 years": thinness_5_9_years,
        "Income Composition of Resources": income_composition_of_resources,
        "Schooling": schooling
    }

    prediction = predict_life_expectancy(input_data)

  
    st.subheader(f"Predicted Life Expectancy: {prediction:.2f} years")