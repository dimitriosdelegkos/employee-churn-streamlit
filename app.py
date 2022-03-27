import streamlit as st
import numpy as np
import pandas as pd
import pickle


head = st.container()
data = st.container()
model_sec = st.container()


@st.cache
def read_data(file):
    return pd.read_csv(file)


@st.cache
def test_prediction(ls):
    """Applying scaling and getting the prediction for a new employee"""
    col_to_be_scaled = ['number_project',
                        'average_montly_hours', 'time_spend_company', 'salary']
    df_pred = pd.DataFrame(columns=['satisfaction_level', 'last_evaluation', 'number_project',
                                    'average_montly_hours', 'time_spend_company', 'Work_accident',
                                    'promotion', 'salary', 'dep_RandD', 'dep_accounting', 'dep_hr',
                                    'dep_management', 'dep_marketing', 'dep_product_mng', 'dep_sales',
                                    'dep_support', 'dep_technical'])
    df_pred.loc[0] = ls
    df_pred.loc[0, col_to_be_scaled] = scaler.transform(
        [df_pred.loc[0, col_to_be_scaled]])
    return model.predict_proba([df_pred.loc[0]])


with head:
    st.title('Employee Churn Prediction')
    st.write(
        'The **goal** of this project is to predict if an employee will leave a company, based on specific features.')
    st.image('churn.jpg')


with data:
    st.header('Employee Dataset')
    st.write('The data includes employee informations, such as the salary, the department, the duration of being at the company, etc.')
    st.write("Let's see the first 5 rows of  the dataset:")
    df = read_data('data/hr_train.csv')
    st.write(df.head())

    with open('data/model', 'rb') as file1, open('data/scaler', 'rb') as file2:
        model = pickle.load(file1)
        scaler = pickle.load(file2)


with model_sec:
    st.header('Model')
    st.write('Here you can select the characteristics of the employee')
    with st.form(key='my_form'):
        col1, col2 = st.columns(2)
        with col1:
            sat_lev = st.slider('Select a satisfaction level',
                                min_value=0.0, max_value=1.0, value=0.2, step=0.01)
            last_eval = st.slider('Select an evaluation level',
                                  min_value=0.0, max_value=1.0, value=0.72, step=0.01)
            proj = st.slider('Select the number of projects',
                             min_value=0, max_value=7, value=6, step=1)
            hours = st.slider('Select the monthly hours',
                              min_value=80, max_value=240, value=224, step=1)
            time_spend = st.slider('Select the years of working',
                                   min_value=0, max_value=10, value=4, step=1)
        with col2:
            accident = st.radio('Had the employee an accident?', ('No', 'Yes'))
            promotion = st.radio(
                'Did the employee get a promotion?', ('No', 'Yes'))
            salary = st.radio('What is the salary?',
                              ('Low', 'Medium', 'High'), index=1)
            department = st.selectbox('Select the department', options=[
                'R&D', 'Accounting', 'HR', 'Management', 'Marketing', 'Product Management', 'Sales', 'Support', 'Technical', 'IT'],
                index=8)
        submit = st.form_submit_button(label='Submit')

        lst = [0] * 17
        if submit:
            lst[0] = sat_lev
            lst[1] = last_eval
            lst[2] = proj
            lst[3] = hours
            lst[4] = time_spend
            if accident == 'No':
                lst[5] = 0
            else:
                lst[5] = 1
            if promotion == 'No':
                lst[6] = 0
            else:
                lst[6] = 1
            if salary == 'Low':
                lst[7] = 0
            elif salary == 'Medium':
                lst[7] = 1
            else:
                lst[7] = 2

            if department == 'R&D':
                lst[8] = 1
            elif department == 'Accounting':
                lst[9] = 1
            elif department == 'HR':
                lst[10] = 1
            elif department == 'Management':
                lst[11] = 1
            elif department == 'Marketing':
                lst[12] = 1
            elif department == 'Product Management':
                lst[13] = 1
            elif department == 'Sales':
                lst[14] = 1
            elif department == 'Support':
                lst[15] = 1
            elif department == 'Technical':
                lst[16] = 1
            else:
                lst[8] = 0

            prediction = test_prediction(lst)
            prediction = np.floor(prediction[0, 1]*100)
            st.write('The probability of leaving the company is:',
                     prediction, '%')
