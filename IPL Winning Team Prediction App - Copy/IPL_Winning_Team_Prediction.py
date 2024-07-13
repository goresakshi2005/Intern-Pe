import streamlit as st 
import pandas as pd
import pickle
import base64

# base 64
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode()
    return base64_string

# Define the teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

base64_image = get_base64_image('D:/IPL Winning Team Prediction App/back2.jpg')

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

# Set the title of the Streamlit app
st.title("IPL Winning Team Prediction App")

st.image("D:/IPL Winning Team Prediction App/IPL.jpg",700,680)

# Read and encode the audio file
st.audio("D:/IPL Winning Team Prediction App/IPL.mp3")


# Load the pre-trained model
model = pickle.load(open('IPL_model.pkl', 'rb'))

# Create columns for the input fields
col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the Batting Team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the Bowling Team', sorted(teams))

city = st.selectbox('Select Host City', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')

with col4:
    overs = st.number_input('Over completed')

with col5:
    wicket = st.number_input('Wicket outs')

# Predict button
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wicket  # Ensure the correct variable name
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    # Create the input DataFrame with correct column names
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wicket': [wickets_left],  # Changed 'wickets' to 'wicket' to match model expectation
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    # Predict probabilities
    result = model.predict_proba(input_df)

    loss = result[0][0]
    win = result[0][1]

    # Display the results
    st.header(batting_team + " - " + str(round(win * 100)) + "%")
    st.header(bowling_team + " - " + str(round(loss * 100)) + "%")
