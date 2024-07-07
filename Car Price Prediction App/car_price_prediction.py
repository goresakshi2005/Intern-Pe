import pandas as pd
import pickle as pk
import streamlit as st
import base64

# base 64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    return base64_string

model= pk.load(open('D:/Car Price Prediction App/car_model.pkl','rb'))

 # Convert local image to base64
base64_image = get_base64_image("D:/Car Price Prediction App/car2.jpg")

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

st.title("Car Price Prediction App")

car= pd.read_csv("D:/Car Price Prediction App/car.csv", encoding= 'latin1')

def get_brand_name(car_name):
    car_name= car_name.split(' ')[0]
    return car_name.strip()
car['name']= car['name'].apply(get_brand_name)

name= st.selectbox("Select Car Brand",car['name'].unique())
year= st.slider("Car manufactured Year",1994,2024)
km_driven= st.slider("No. of KMs Driven",11,200000)
fuel= st.selectbox("Fuel Type",car['fuel'].unique())
seller_type= st.selectbox("Seller Type",car['seller_type'].unique())
transmission= st.selectbox("Transmission Type",car['transmission'].unique())
owner= st.selectbox("Owner",car['owner'].unique())
mileage= st.slider("Car Mileage",10,40)
engine= st.slider("Engine CC",700,5000)
max_power= st.slider("Max Power",0,200)
seats= st.slider("No. of Seats",5,10)


if st.button("Predict"):
    input_data_model= pd.DataFrame([[name,year,km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats]],columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])
    
    input_data_model['owner'].replace(['First Owner', 'Second Owner', 'Third Owner','Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5], inplace= True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4],inplace= True)
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3],inplace= True)
    input_data_model['transmission'].replace(['Manual', 'Automatic'],[1,2],inplace= True)
    input_data_model['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], inplace= True)
   
    car_price= model.predict(input_data_model)
    
    st.markdown(f"Car Price is going to be: {car_price[0]}")


    
    














