import streamlit as st
import pickle
import numpy as np
from src.utils import *

model = load_model('saves/models/Voting Classifier_20241129_040510.pkl')
def predict(input_data):
    
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    if prediction == 1:
        return "fraud!"
    else:
        return "legit transaction"
    

# Create the Streamlit app
st.title('Credit card fraud detection Model Deployment with Streamlit')
st.write('Enter the input features to get a prediction.')
feature1 = st.number_input('Time', value=0.0)
feature2 = st.number_input('V1', value=0.0)
feature3 = st.number_input('V2', value=0.0)
feature4 = st.number_input('V3', value=0.0)

feature5 = st.number_input('V4', value=0.0)
feature6 = st.number_input('V5', value=0.0)
feature7 = st.number_input('V6', value=0.0)
feature8 = st.number_input('V7', value=0.0)

feature9 = st.number_input('V8', value=0.0)
feature10 = st.number_input('V9', value=0.0)
feature11 = st.number_input('V10', value=0.0)
feature12 = st.number_input('V11', value=0.0)

feature13 = st.number_input('V12', value=0.0)
feature14 = st.number_input('V13', value=0.0)
feature15 = st.number_input('V14', value=0.0)
feature16 = st.number_input('V15', value=0.0)

feature17 = st.number_input('V16', value=0.0)
feature18 = st.number_input('V17', value=0.0)
feature19 = st.number_input('V18', value=0.0)
feature20 = st.number_input('V19', value=0.0)

feature21 = st.number_input('V20', value=0.0)
feature22 = st.number_input('V21', value=0.0)
feature23 = st.number_input('V22', value=0.0)
feature24 = st.number_input('V23', value=0.0)

feature25 = st.number_input('V24', value=0.0)
feature26 = st.number_input('V25', value=0.0)
feature27 = st.number_input('V26', value=0.0)
feature28 = st.number_input('V27', value=0.0)
feature29 = st.number_input('V28', value=0.0)
feature30 = st.number_input('Amount', value=0.0)

if st.button('Predict'):
    input_data = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10,feature11, feature12, feature13, feature14,feature15, feature16, feature17, feature18, feature19, 
                  feature20, feature21, feature22, feature23, feature24, feature25, feature26, feature27, feature28, feature29, feature30]
    prediction = predict(input_data)
    st.write(f'{prediction}')

