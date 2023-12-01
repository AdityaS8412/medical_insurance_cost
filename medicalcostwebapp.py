import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def medical_prediction(input_data):
    

    #changing input_data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    #reshaping the array
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print("The inasurence cost in USD" , prediction[0])
    
    
def main():
    #giving a title
    st.title('Medical Cost Prediction Web App')
    
    #getting the inout data from the user
    
    age=st.text_input('Age of person:')
    
    sex=st.text_input('Sex of person:')
    
    bmi=st.text_input('BMI of person:')
    
    smoker=st.text_input('Smoker or not:')
    
    region=st.text_input('Region of person:')
    
    #code for prediction
    cost=''
    
    #creating a button for prediction
    if st.button('Medical cost:'):
        cost= medical_prediction([age,sex,bmi,smoker,region])
        
    st.success(cost) 
    





if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    