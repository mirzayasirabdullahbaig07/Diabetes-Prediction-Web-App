# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 18:23:13 2025

@author: HP
"""

import numpy as np
import pickle
import streamlit as st


#loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))



# creating a function

def diabetes_prediction(input_data):
    # Load scaler and model


    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    
    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return    'The person is diabetic'
  
    
  
    
def main():
    #giving a name
    st.title('Diabetes Prediction Web App')
    
    # getting the input data 
    Pregnancies = st.text_input('Numbers of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Bloodpressure value')
    SkinThinkness = st.text_input('SkinThinkness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of person')
    
    
    # code for prediction
    diagnosis = ''      
    
    
    # creating a button for prediction
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([
    float(Pregnancies), float(Glucose), float(BloodPressure),
    float(SkinThinkness), float(Insulin), float(BMI),
    float(DiabetesPedigreeFunction), float(Age)])
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
