
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle

def load_pickle(pickle_file):
    loaded_pickle = joblib.load(open(os.path.join(pickle_file), 'rb'))
    return loaded_pickle

def run_promotion_evaluator():
    # st.subheader('ML Prediction Section')

    st.subheader('Input Your Data')
    department = st.selectbox('Department', ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D',
                                              'Procurement', 'Finance', 'HR', 'Legal'])
    region = st.selectbox('Region', ['region_1', 'region_2', 'region_3', 'region_4','region_5', 'region_6', 'region_7',
                                     'region_8', 'region_9', 'region_10', 'region_11', 'region_12', 'region_13', 'region_14', 
                                     'region_15', 'region_16', 'region_17', 'region_18', 'region_19', 'region_20', 'region_21', 
                                     'region_22', 'region_23', 'region_24', 'region_25', 'region_26', 'region_27', 'region_28', 
                                     'region_29', 'region_30', 'region_31', 'region_32', 'region_33', 'region_34'])
    education = st.selectbox('Education', ['Below Secondary', "Bachelor's", "Master's & above" ])
    gender = st.radio('Gender', ['m', 'f'])
    recruitment = st.selectbox('Recruitment Channel', ['referred', 'sourcing', 'Others'])
    training = st.number_input('No of Training', 1, 10)
    age = st.number_input('Age', 10, 60)
    rating = st.number_input('Previous Year Rating', 1, 5)
    service = st.number_input('Length Of Service', 1, 37)
    awards = st.radio('Awards Won', [0, 1])
    avg_training = st.number_input('Average Training Score', 0, 100)

    with st.expander('Your Selected Options'):
        result = {
            'Department': department,
            'Region': region,
            'Education': education,
            'gender': gender,
            'Recruitment Channel': recruitment,
            'No of Training': training,
            'Age': age,
            'Previous Year Rating': rating,
            'Length of Service': service,
            'Awards Won': awards,
            'Average Training Score': avg_training,
        }
    st.write(result)

    label_encoder_department = load_pickle('a_label_encoder_department.pkl')
    label_encoder_region = load_pickle('a_label_encoder_region.pkl')
    label_encoder_education = load_pickle('a_label_encoder_education.pkl')
    label_encoder_gender = load_pickle('a_label_encoder_gender.pkl')
    label_encoder_recruitment_channel = load_pickle('a_label_encoder_recruitment_channel.pkl')

    encoded_result = []

    for i in result.values():
        if type(i) == int:
            encoded_result.append(i)
        if i in ['Sales & Marketing', 'Operations', 'Technology', 'Analytics', 'R&D',
                                              'Procurement', 'Finance', 'HR', 'Legal']:
            encoded_result.append(label_encoder_department.transform([i])[0])
        elif i in ['region_1', 'region_2', 'region_3', 'region_4','region_5', 'region_6', 'region_7',
                                     'region_8', 'region_9', 'region_10', 'region_11', 'region_12', 'region_13', 'region_14', 
                                     'region_15', 'region_16', 'region_17', 'region_18', 'region_19', 'region_20', 'region_21', 
                                     'region_22', 'region_23', 'region_24', 'region_25', 'region_26', 'region_27', 'region_28', 
                                     'region_29', 'region_30', 'region_31', 'region_32', 'region_33', 'region_34']:
            encoded_result.append(label_encoder_region.transform([i])[0])
        elif i in ['Below Secondary', "Bachelor's", "Master's & above" ]:
            encoded_result.append(label_encoder_education.transform([i])[0])
        elif i in ['m', 'f']:
            encoded_result.append(label_encoder_gender.transform([i])[0])
        elif i in ['referred', 'sourcing', 'Others']:
            encoded_result.append(label_encoder_recruitment_channel.transform([i])[0])

    st.write(encoded_result)

    model = load_pickle('1_model_grad.pkl')

    reshaped_result = np.array(encoded_result).reshape(1, -1)
    prediction = model.predict(reshaped_result)

    if prediction == 1:
        st.success(100*' ||||CONGRATULATION, YOU GET PROMOTION||||')
    else:
        st.warning(100*' ||||SORRY, WORK HARDER||||')