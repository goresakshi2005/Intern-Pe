import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import streamlit as st
import base64


#loading the saved model
loaded_model= pickle.load(open("D:/Breast Cancer Detection App/breast_model.sav",'rb'))

# base 64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    return base64_string

# creating a function for prediction
def Breast_cancer_detection(input_data):
     
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
        return"The Breast Cancer is Malignant"
    else:
        return"The Breast Cancer is Benign"

def main():
     # Convert local image to base64
    base64_image = get_base64_image("D:/Breast Cancer Detection App/back2.jpg")
    
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
    st.title("Breast Cancer Detection App")
    st.subheader("Enter all necessary features")
    # st.image('D:/Diabetes Prediction App/doctor.png', width=700)
    # getting the input from user
    radius_mean= st.text_input("radius_mean")
    texture_mean= st.text_input("texture_mean")
    perimeter_mean= st.text_input("perimeter_mean")
    area_mean= st.text_input("area_mean")
    smoothness_mean= st.text_input("smoothness_mean")
    compactness_mean= st.text_input("compactness_mean")
    concavity_mean= st.text_input("concavity_mean")
    concave_points_mean= st.text_input("concave_points_mean")
    symmetry_mean= st.text_input("symmetry_mean")
    fractal_dimension_mean= st.text_input("fractal_dimension_mean")
    radius_se= st.text_input("radius_se")
    texture_se= st.text_input("texture_se")
    perimeter_se= st.text_input("perimeter_se")
    area_se= st.text_input("area_se")
    smoothness_se= st.text_input("smoothness_se")
    compactness_se= st.text_input("compactness_se")
    concavity_se= st.text_input("concavity_se")
    concave_points_se= st.text_input("concave_points_se")
    symmetry_se= st.text_input("symmetry_se")
    fractal_dimension_se= st.text_input("fractal_dimension_se")
    radius_worst= st.text_input("radius_worst")
    texture_worst= st.text_input("texture_worst")
    perimeter_worst= st.text_input("perimeter_worst")
    area_worst= st.text_input("area_worst")
    smoothness_worst= st.text_input("smoothness_worst")
    compactness_worst= st.text_input("compactness_worst")
    concavity_worst= st.text_input("concavity_worst")
    concave_points_worst= st.text_input("concave_points_worst")
    symmetry_worst= st.text_input("symmetry_worst")
    fractal_dimension_worst= st.text_input("fractal_dimension_worst")


    # code for prediction
    diagnosis= ""
    # creating the button
    if st.button("Breast Cancer Test Result"):
        diagnosis= Breast_cancer_detection([radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst])
    
    st.success(diagnosis)
    
if __name__=='__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    