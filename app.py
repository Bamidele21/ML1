import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the trained model
model = tf.keras.models.load_model('ann_model.h5')

## Load the encoders and scaler
with open('label_encoded_gender.pkl', 'rb') as file:
    label_encoded_gender = pickle.load(file)

with open('onehot_encoded_geo.pkl', 'rb') as file:
    onehot_encoded_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
##streamlit app

st.title( 'Customer Churn Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoded_geo.categories_[0])
gender = st.selectbox('Gender', label_encoded_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
num_of_products = st.number_input('Number of Products')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
number_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.checkbox('Has Credit Card')
is_active_member = st.checkbox('Is Active Member')

##prepare the input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoded_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})
##OHE encode 'Geography'
geo_encoded = onehot_encoded_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoded_geo.get_feature_names_out(['Geography']))

## Combine one-hot encoded columns with input data

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

scaled_input_data = scaler.transform(input_data)

## Predict churn

prediction = model.predict(scaled_input_data)

probability = prediction[0][0]

if probability > 0.5:
    st.write('Customer is likely to churn' )

else:
    st.write('Customer is not likely to churn')