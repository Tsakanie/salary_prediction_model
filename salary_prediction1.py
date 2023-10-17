import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.title("Salary Prediction page")
data=pd.read_csv("C:/Users/malwandlats/OneDrive - SABC Pty Ltd/Documents/Data Science/web development/Salary_Data.csv")
x=data["YearsExperience"]
st.write("X--> Years of Experience")
y=data["Salary"]
st.write("Y--> Salary")
m=st.sidebar.radio("Menu",["EDA","Prediction"])
if m=='EDA':
    st.subheader("EDA")
    st.write("Shape")
    st.write(data.shape)
    st.write("Head")
    st.write(data.head())
    st.write("Dataset info")
    st.write(data.describe())
    fig, ax=plt.subplots(figsize=(10,5))
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.title("Years of experience VS Salary")
    plt.scatter(x,y)
    st.pyplot(fig)
elif m=="Prediction":
    st.subheader("Prediction")
    from sklearn.model_selection import train_test_split
    x_train,x_text,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    x=np.array(x).reshape(-1,1)
    y=np.array(y).reshape(-1,1)
    from sklearn.linear_model import LinearRegression
    regressor=LinearRegression()
    regressor.fit(x,y)
    import pickle
    r=open("regression.pkl","wb")
    pickle.dump(regressor,r)
    r.close()
    exp=st.number_input("Enter yor experience in years",0,42,5)
    exp=np.array(exp).reshape(1,-1)
    prediction=regressor.predict(exp)[0]
    if st.button("Salary Prediction"):
        st.write(f"{prediction}")

