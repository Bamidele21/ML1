import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf 
import streamlit as st 
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime
import streamlit as st
##fetch the data from the csv file 
data = pd.read_csv('Churn_Modelling.csv')



##preprocessing the data ---------
##drop irrelavant columns 
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)



##encode categorical data to ones and zeros 

label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])

#use OneHotEncoder to encode categorical variables over 2 
geo_onehot_encoder=OneHotEncoder()
geo_encoder=geo_onehot_encoder.fit_transform(data[['Geography']])

geo_array = geo_onehot_encoder.get_feature_names_out(['Geography'])

geo_encodedpf = pd.DataFrame(geo_encoder.toarray(), columns=geo_array)



##combine OHE columns with the original data 

data = pd.concat([data.drop('Geography', axis=1), geo_encodedpf], axis=1)



##save the encoders and scaler to a pickle file

##Divide the dataset into independent and dependent features 

X = data.drop('Exited', axis=1)
y = data['Exited']


## Split the data in training and tetsing sets

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Scale these features

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(x_test)

#ANN model training

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), ## Hidden Layer 1 connected with input layer
    Dense(32, activation='relu'), ## Hidden Layer 2
    Dense(1, activation='sigmoid')  ## Output Layer
])

model_sum = model.summary()



##copiling the model 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

##Set up the TensorBoard callback

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensor_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

##setting up early stopping callback

early_stopping_callback=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

##training the model
history= model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, callbacks=[tensor_callback, early_stopping_callback])





##  Making predictions with the trained model using the test data -----------------------------------
## Load the trained model, scaler and encoders from the pickle file


model = load_model('ann_model.h5')


##load encoder and scaler

with open('label_encoded_gender.pkl', 'rb') as file:
    label_encoded_gender = pickle.load(file)

with open('onehot_encoded_geo.pkl', 'rb') as file:
    onehot_encoded_geo = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
    ## Example input data
    
    input_data = {
        'CreditScore': 600,
        'Geography': 'France',
        'Gender': 'Male',
        'Age': 40,
        'Tenure': 3,
        'Balance': 60000,
        'NumOfProducts': 2,
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'EstimatedSalary': 50000
    }
    input_df = pd.DataFrame([input_data])
    
    onehot_encoded_geo_input = onehot_encoded_geo.transform([[input_data['Geography']]]).toarray()
    geo_encoded_df = pd.DataFrame(onehot_encoded_geo_input, columns=onehot_encoded_geo.get_feature_names_out(['Geography']))
    
   
    

##enode  catogorical variables 

input_df['Gender']= label_encoded_gender.transform(input_df['Gender'])

## concatenate one-hot encoded columns with the original data

input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)

scale_input = scaler.transform(input_df)

## Predict churn

prediction = model.predict(scale_input)

probability = prediction[0][0]

if probability > 0.5:
   print('The customer is likely to churn.')
   
else:
    print('The customer is not likely to churn.')
    