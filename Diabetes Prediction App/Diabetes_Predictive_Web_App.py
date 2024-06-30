import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64


#loading the saved model
loaded_model= pickle.load(open("D:/Diabetes Prediction App/trained_model.sav",'rb'))

# base 64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    return base64_string

# creating a function for prediction
def diabetes_prediction(input_data):
     
    # changing the input to numpy array
    input_as_nparray= np.asarray(input_data)
    # reshape the array as we are predicting for one instance
    input_reshaped= input_as_nparray.reshape(1,-1)
    # standardize the input data
    scaler= StandardScaler()
    scaler.fit(input_reshaped)
    std_data= scaler.transform(input_reshaped)
    prediction= loaded_model.predict(std_data)

    if(prediction[0]== 0):
        return"The Person is not Diabetic"
    else:
        return"The Person is Diabetic"

def main():
     # Convert local image to base64
    base64_image = get_base64_image("D:/Diabetes Prediction App/back.jpg")
    
    # Set background image using CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("data:image/jpg;base64,{base64_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # giving a title
    st.title("Diabetes Prediction Web App")
    st.image('D:/Diabetes Prediction App/doctor.png', width=700)
    # getting the input from user
    Pregnancies= st.text_input("Number of Pregnancies")
    Glucose= st.text_input("Glucose Level")
    BloodPressure= st.text_input("Blood Pressure Value")
    SkinThickness= st.text_input("Skin Thickness Value")
    Insulin= st.text_input("Insulin Level")
    BMI= st.text_input("BMI Value")
    DiabetesPedigreeFunction= st.text_input("Diabetes Pedigree Function Value")
    Age= st.text_input("Age of the Person")
    
    # code for prediction
    diagnosis= ""
    # creating the button
    if st.button("Diabetes Test Result"):
        diagnosis= diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
if __name__=='__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    