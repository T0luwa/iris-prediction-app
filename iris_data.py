import streamlit as st
import sklearn
import pickle
import pandas as pd
from PIL import Image
image = Image.open("iris.webp")
#image = Image.open("/Users/test/Downloads/iris.webp") to show image offline
st.image(image, width = 350)
iris_data = pickle.load(open('randomforest.sav', 'rb'))

st.title('Iris Data prediction app')

def user_report():
    sepal_length = st.sidebar.slider('sepal_length', 4.3, 7.9, 1.0)
    sepal_width = st.sidebar.slider('sepal_width', 2.0, 4.4, 1.0)
    petal_length = st.sidebar.slider('petal_length', 1.0, 6.9, 1.0)
    petal_width = st.sidebar.slider('petal_width', 0.1, 2.5, 1.0)

    user_report = {
        'sepal.length': sepal_length,
        'sepal.width': sepal_width,
        'petal.length': petal_length,
        'petal.width': petal_width
    }
    user_report_data = pd.DataFrame(user_report, index=[0])
    return user_report_data

user_data = user_report()

st.header('iris data')

st.write(user_data)

prediction = iris_data.predict(user_data)

if (prediction == 0):
    st.success('setosa')
elif (prediction == 1):
    st.success('vesicolor')
else:
    st.success('virginica')
