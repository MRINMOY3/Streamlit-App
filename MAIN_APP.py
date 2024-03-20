from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import plotly.express as px
import warnings
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as com
from plotly import graph_objs as obj
import os as os

warnings.filterwarnings('ignore')




st.set_page_config(page_title="Sixth Sense",page_icon=":medical_symbol:",layout="wide")

st.title(":medical_symbol: Sixth Sense | A medical DATA ANALYZER AND DAILY HEALTH PREDICTOR")


st.markdown('<style>div.block-container{padding-top:1rem}</style>',unsafe_allow_html=True)

fl =st.file_uploader(":file_folder: Upload A file",type=(["csv","txt","xlsx","xls"]))

if fl is not None:
    filename = fl.name
    st.write(filename)
    df = pd.read_csv("filename")## Reading th csv file
    ## We will read the csv file over here 
else:
    os.chdir(r"C:\Users\KIIT\PYTHON\ML_DS_MINOR_PROJECT\A1")
    df = pd.read_csv("data//Sleep_Health_Diseases.csv",encoding="ISO-8859-1")




st.title("INPUT FORM FOR YOUR DAILY HEALTH STATS")
first,last  =  st.columns(2)

first.text_input("First Name")
last.text_input("Last Name")


email,mob = st.columns([3,1])
email.text_input("Enter Your Email📩")
email.text_input("Enter Your Mobile📲number")


user,pw,pw2 = st.columns(3)
user.text_input("Username Please")
pw.text_input("Enter Your Password",type="password")
pw2.text_input("Enter Confirmed Password",type="password") 




ch,bl,sub = st.columns(3)
ch.checkbox("I Agree")
sub.button("Submit")




st.title("User Data Input Form")

# Form fields
person_id = st.number_input("Person ID", min_value=1, step=1, value=1)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=0, step=1, value=27)
occupation = st.text_input("Occupation", "Software Engineer")
sleep_duration = st.number_input("Sleep Duration (hours)", min_value=0.0, value=6.1, step=0.1)
quality_of_sleep = st.slider("Quality of Sleep", min_value=0, max_value=10, value=6, step=1)
physical_activity_level = st.number_input("Physical Activity Level", min_value=0, step=1, value=42)
stress_level = st.slider("Stress Level", min_value=0, max_value=10, value=6, step=1)
bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
blood_pressure = st.text_input("Blood Pressure", "126/83")
heart_rate = st.number_input("Heart Rate", min_value=0, step=1, value=77)
daily_steps = st.number_input("Daily Steps", min_value=0, step=100, value=4200)
sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Insomnia", "Sleep Apnea", "Restless Legs Syndrome"])

# Submit button
submitted = st.button("Submit Your Daily Medical Statistics Here")

if submitted:
    st.success("Form submitted successfully!")
    # Display submitted data in a table
    st.header("Submitted User Data")
    data = {
        "Person ID": person_id,
        "Gender": gender,
        "Age": age,
        "Occupation": occupation,
        "Sleep Duration (hours)": sleep_duration,
        "Quality of Sleep": quality_of_sleep,
        "Physical Activity Level": physical_activity_level,
        "Stress Level": stress_level,
        "BMI Category": bmi_category,
        "Blood Pressure": blood_pressure,
        "Heart Rate": heart_rate,
        "Daily Steps": daily_steps,
        "Sleep Disorder": sleep_disorder
    }
    st.table(data)
    # Analysis options

    df = pd.DataFrame([data])

    st.sidebar.header("The user Data Inputted")
    analysis_option1  = st.sidebar.radio("Select Analysis for User", ("Display in Histogram form", "Display in Scatter Plot form"))
    if analysis_option1 == "Display in Histogram form":
       st.subheader("Histogram")
       st.write("Select a feature to display its histogram:")
       selected_feature = st.selectbox("Select Feature", df.columns)
       plt.hist(df[selected_feature], bins=10)
       st.pyplot(plt)
   
    elif analysis_option1 == "Display in Scatter Plot form":
       st.subheader("Scatter Plot")
       st.write("Select two features to display their scatter plot:")
       x_feature = st.selectbox("X Feature", df.columns)
       y_feature = st.selectbox("Y Feature", df.columns)
       plt.scatter(df[x_feature], df[y_feature])
       plt.xlabel(x_feature)
       plt.ylabel(y_feature)
       st.pyplot(plt)



st.sidebar.header("Data Analysis")
analysis_option = st.sidebar.radio("Select Analysis", ("Histogram", "Scatter Plot"))

    # Dataframe creation
    

    # Perform analysis based on user selection
if analysis_option == "Histogram":
    st.subheader("Histogram")
    st.write("Select a feature to display its histogram:")
    selected_feature = st.selectbox("Select Feature", df.columns)
    plt.hist(df[selected_feature], bins=10)
    st.pyplot(plt)












elif analysis_option == "Scatter Plot":
    st.subheader("Scatter Plot")
    st.write("Select two features to display their scatter plot:")
    x_feature = st.selectbox("X Feature", df.columns)
    y_feature = st.selectbox("Y Feature", df.columns)
    plt.scatter(df[x_feature], df[y_feature])
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    st.pyplot(plt)
# df["Order Date"] = pd.to_datetime(df["Order Date"])



# ## Getting the min and max data from the column

# startDate = pd.to_datetime(df["Order Date"]).min()
# endDate = pd.to_datetime(df["Order Date"]).max()

# with col1:
#     date1 = pd.to_datetime(st.date_input("Start Date",startDate))

# with col2:
#     date2 = pd.to_datetime(st.date_input("End Date",endDate))



# df = df[(df["Order Date"]>=date1 )  & (df["Order Date"]<=date2)].copy()



data = pd.read_csv("data//MHD.csv")

nav = st.sidebar.radio("NAVIGATION PANEL",["Home","Predict Your Records","Analyze"])

if nav == "Home":
    # st.image("data//put.jpg")
    st.write("Home")        

    if(st.checkbox("Show Table | STATISTICAL ANALYSIS")):
        dt = pd.read_csv("data//Sleep_Health_Diseases.csv")
        dt_json = dt.to_json(orient="records")
        st.dataframe(dt)
    
    if(st.checkbox("Show the detailed Height ANALYSIS\n")):
        height = np.random.normal(140,20,500)
        plt.hist(height,bins=100,ec='white')
        st.pyplot(plt)


        # st.table(data)

        graph = st.selectbox("What Kind of GRAPH ?",["Non_Interactive","Interactive"])
        if graph == "Non_Interactive":
            plt.figure(figsize=(10,5))
            a1 = np.random.randint(1,100,50)
            a2 = a1*2

            plt.scatter(a1,a2)
            st.pyplot(plt)

        if graph == "Interactive":
            pass


if nav == "Predict Your Records":
    st.write("Predicted Values")

if nav == "Analyze":
    st.write("Statistical Analysis of Your data")


















rad=st.sidebar.radio("Navigation Menu",["Home","Covid-19","Diabetes","Heart Disease","Plots"])

#Home Page 

#displays all the available disease prediction options in the web app
if rad=="Home":
    st.title("Medical Predictions App")
    st.text("The Following Disease Predictions Are Available ->")
    st.text("1. Covid-19 Infection Predictions")
    st.text("2. Diabetes Predictions")
    st.text("3. Heart Disease Predictions")

#Covid-19 Prediction

#loading the Covid-19 dataset
df1=pd.read_csv("data//Covid-19 Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x1) & target(y1)
x1=df1.drop("Infected with Covid19",axis=1)
x1=np.array(x1)
y1=pd.DataFrame(df1["Infected with Covid19"])
y1=np.array(y1)
#performing train-test split on the data
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model1=RandomForestClassifier()
#fitting the model with train data (x1_train & y1_train)
model1.fit(x1_train,y1_train)

#Covid-19 Page

#heading over to the Covid-19 section
if rad=="Covid-19":
    st.header("Know If You Are Affected By Covid-19")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Dry Cough (drycough), Fever (fever), Sore Throat (sorethroat), Breathing Problem (breathingprob)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    drycough=st.number_input("Rate Of Dry Cough (0-20)",min_value=0,max_value=20,step=1)
    fever=st.number_input("Rate Of Fever (0-20)",min_value=0,max_value=20,step=1)
    sorethroat=st.number_input("Rate Of Sore Throat (0-20)",min_value=0,max_value=20,step=1)
    breathingprob=st.number_input("Rate Of Breathing Problem (0-20)",min_value=0,max_value=20,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction1=model1.predict([[drycough,fever,sorethroat,breathingprob]])[0]
    
    #prediction part predicts whether the person is affected by Covid-19 or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction1=="Yes":
            st.warning("You Might Be Affected By Covid-19")
        elif prediction1=="No":
            st.success("You Are Safe")

#Diabetes Prediction

#loading the Diabetes dataset
df2=pd.read_csv("data//Diabetes Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x2) & target(y2)
x2=df2.iloc[:,[1,4,5,7]].values
x2=np.array(x2)
y2=y2=df2.iloc[:,[-1]].values
y2=np.array(y2)
#performing train-test split on the data
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model2=RandomForestClassifier()
#fitting the model with train data (x2_train & y2_train)
model2.fit(x2_train,y2_train)

#Diabetes Page

#heading over to the Diabetes section
if rad=="Diabetes":
    st.header("Know If You Are Affected By Diabetes")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Glucose (glucose), Insulin (insulin), Body Mass Index-BMI (bmi), Age (age)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    glucose=st.number_input("Enter Your Glucose Level (0-200)",min_value=0,max_value=200,step=1)
    insulin=st.number_input("Enter Your Insulin Level In Body (0-850)",min_value=0,max_value=850,step=1)
    bmi=st.number_input("Enter Your Body Mass Index/BMI Value (0-70)",min_value=0,max_value=70,step=1)
    age=st.number_input("Enter Your Age (20-80)",min_value=20,max_value=80,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction2=model2.predict([[glucose,insulin,bmi,age]])[0]
    
    #prediction part predicts whether the person is affected by Diabetes or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if prediction2==1:
            st.warning("You Might Be Affected By Diabetes")
        elif prediction2==0:
            st.success("You Are Safe")

#Heart Disease Prediction

#loading the Heart Disease dataset
df3=pd.read_csv("data//Heart Disease Predictions.csv")
#cleaning the data by dropping unneccessary column and dividing the data as features(x3) & target(y3)
x3=df3.iloc[:,[2,3,4,7]].values
x3=np.array(x3)
y3=y3=df3.iloc[:,[-1]].values
y3=np.array(y3)
#performing train-test split on the data
x3_train,x3_test,y3_train,y3_test=train_test_split(x3,y3,test_size=0.2,random_state=0)
#creating an object for the model for further usage
model3=RandomForestClassifier()
#fitting the model with train data (x3_train & y3_train)
model3.fit(x3_train,y3_train)

#Heart Disease Page

#heading over to the Heart Disease section
if rad=="Heart Disease":
    st.header("Know If You Are Affected By Heart Disease")
    st.write("All The Values Should Be In Range Mentioned")
    #taking the 4 most important features as input as features -> Chest Pain (chestpain), Blood Pressure-BP (bp), Cholestrol (cholestrol), Maximum HR (maxhr)
    #a min value (min_value) & max value (max_value) range is set so that user can enter value within that range
    #incase user enters a value which is not in the range then the value will not be taken whereas an alert message will pop up
    chestpain=st.number_input("Rate Your Chest Pain (1-4)",min_value=1,max_value=4,step=1)
    bp=st.number_input("Enter Your Blood Pressure Rate (95-200)",min_value=95,max_value=200,step=1)
    cholestrol=st.number_input("Enter Your Cholestrol Level Value (125-565)",min_value=125,max_value=565,step=1)
    maxhr=st.number_input("Enter You Maximum Heart Rate (70-200)",min_value=70,max_value=200,step=1)
    #the variable prediction1 predicts by the health state by passing the 4 features to the model
    prediction3=model3.predict([[chestpain,bp,cholestrol,maxhr]])[0]
    
    #prediction part predicts whether the person is affected by Heart Disease or not by the help of features taken as input
    #on the basis of prediction the results are displayed
    if st.button("Predict"):
        if str(prediction3)=="Presence":
            st.warning("You Might Be Affected By Diabetes")
        elif str(prediction3)=="Absence":
            st.success("You Are Safe")
                                        
#Plots Page

#heading over to the plots section
#plots are displayed for each disease prediction section 
if rad=="Plots":
    #
    type=st.selectbox("Which Plot Do You Want To See?",["Covid-19","Diabetes","Heart Disease"])
    if type=="Covid-19":
        fig=px.scatter(df1,x="Difficulty in breathing",y="Infected with Covid19")
        st.plotly_chart(fig)

    elif type=="Diabetes":
        fig=px.scatter(df2,x="Glucose",y="Outcome")
        st.plotly_chart(fig)
    elif type=="Heart Disease":
        fig=px.scatter(df3,x="BP",y="Heart Disease")
        st.plotly_chart(fig)
