import streamlit as st
import os
import numpy as np 
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import sklearn.metrics as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



os.chdir('C:\\Users\\danie\\Documents\\Final')
s = pd.read_csv("social_media_usage.csv")



def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] > 9, np.nan,s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan, s["educ2"]),
    "parent":np.where(s["par"] > 2, np.nan, s["par"]),
    "married":np.where(s["marital"] == 8|9 , np.nan, np.where(s["marital"] > 1, 2, 1)),
    "female":np.where(s["gender"] == 3|8|9 , np.nan, np.where(s["gender"] > 1, 2, 1)),
    "age":np.where(s["age"] > 98, np.nan, s["age"])}).dropna()

print(ss)


y = ss["sm_li"]
X = ss[["income", "education", "parent", "married","female","age"]]



X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.2,    
                                                    random_state=987) 

lr = LogisticRegression(class_weight = "balanced")


lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)


newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female": [1, 1],
    "age": [42, 82],
})



newdata["sm_li"] = lr.predict(newdata)


person_1 = [8,7,0,1,1,42]

person_2 = [8,7,0,1,1,82]

predicted_class_1 = lr.predict([person_1])
predicted_class_2 = lr.predict([person_2])


probs_1 = lr.predict_proba([person_1])
probs_2 = lr.predict_proba([person_2])



st.markdown("# Welcome to my prediction app!")

st.markdown("### Please enter your answers below to see if we can correctly predict if you are a Linkedin User!")



num1 = st.slider(label="Enter your Income (low = 1 to high =9)", 
           min_value=1,
           max_value=9,
           value=2)

num2 = st.slider(label="Enter your Education level (Less than highschool = 1 to Postgraduate degree = 8)", 
           min_value=1,
           max_value=8,
           value=2)

num3 = st.slider(label="Are you a parent? (Yes = 1 to No = 2)", 
           min_value=1,
           max_value=2,
           value=2)

num4 = st.slider(label="Are you married? (Yes = 1 to No = 2)", 
           min_value=1,
           max_value=2,
           value=2)

num5 = st.slider(label="What is your gender? (Female = 1 to Male = 2)", 
           min_value=1,
           max_value=2,
           value=2)

num6 = st.slider(label="How old are you?", 
           min_value=1,
           max_value=97,
           value=10)

new_person = [num1,num2,num3,num4,num5,num6]

predicted_class_new = lr.predict([new_person])

probs_new= lr.predict_proba([new_person])

# Linkedin User
if predicted_class_new == 1:
    inc_label = "Yes"
else:
    inc_label = "No"


prob_label = probs_new.round(2)

st.write(f"Predicted Linkedin User: {inc_label}, with a probability of {prob_label[0][1]}")


fig = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = prob_label[0][1],
    title = {'text': f"Linkedin User: {inc_label}"},
    gauge = {"axis": {"range": [0, 1]},
            "steps": [
                {"range": [0, .3], "color":"red"},
                {"range": [.3, .6], "color":"gray"},
                {"range": [.6, 1], "color":"lightgreen"}
            ],
            "bar":{"color":"yellow"}}
    ))

st.plotly_chart(fig)