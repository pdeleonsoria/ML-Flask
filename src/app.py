
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Cargar el modelo
with open('/workspace/ML-Flask/models/modelo_reg_log.pkl', 'rb') as file:
    model = pickle.load(file)

# Título
st.title("🚢 ¿Sobrevivirías al Titanic? 🚢")

# Formulario
with st.form(key='titanic_form'):
    sex = st.selectbox("Sexo:", ["Hombre", "Mujer"])
    fare = st.slider("Cuánto pagarías por el boleto", min_value=0, max_value=512, value=35, step=1)
    familiares = st.number_input("Número de familiares:", min_value=0)
    age = st.number_input("Edad:", min_value=0, max_value=80)
    pclass = st.selectbox("Clase del boleto:", [1, 2, 3])
    embarked = st.selectbox("¿Desde qué puerto saldrías?", ["Southampton (UK)", "Queenstown (IR)", "Cherbourg (FR)"])

    # Convertir a numérico
    sex_n = 1 if sex == "Mujer" else 0

    embarked_n = 0 if embarked == "Southampton (UK)" else 1 if embarked == "Cherbourg (FR)" else 2
    
    # Crear un DataFrame con los datos ingresados, en el mismo orden que se utilizó para entrenar el modelo porque si no me daba error 
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'Fare': [fare],
        'Sex_n': [sex_n],
        'Embarked_n': [embarked_n],
        'Familiares': [familiares]
    })

    # Botón 
    submit_button = st.form_submit_button("Predecir Supervivencia")

if submit_button:

    #PRedecir 
    prediction = model.predict(input_data)

    # Resultado
    if prediction[0] == 1:
        st.success("🥳¡Sobrevives!🥳")
    else:
        st.error("☠️ No sobrevives.☠️")