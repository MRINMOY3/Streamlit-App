import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


numericvals = []
# Function to preprocess the data
def preprocess_data(data):
    # Handle missing values
    data.fillna(data.mean(), inplace=True)  # Fill missing values with mean for numeric columns

    # Encode categorical variables
    label_encoders = {}
    for column in ['Gender', 'BMI Category']:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])
    
    return data
  
# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Drop irrelevant columns
    data.drop(['Person ID', 'Occupation', 'Blood Pressure'], axis=1, inplace=True)

    # Preprocess the data
    data = preprocess_data(data)
    
    return data

# Function to train the model
def train_model(X_train, y_train):
    # Train a Random Forest classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

# Function to make predictions
def predict(model, X_test):
    # Make predictions
    y_pred = model.predict(X_test)
    return y_pred

# Main function to run the Streamlit app
def main():
    st.sidebar.title("NAVIGATION PANEL")
    nav = st.sidebar.radio("Navigation", ["Home", "See the attributes in your RECORD :-", "See the Entries of your data Set","Enter details of a person","Data Analysis"])
    
    data = pd.read_csv("data//Sleep_Health_Diseases.csv")

    if nav == "See the attributes in your RECORD :-":
        st.write("Column Names:", data.columns.tolist())

    elif nav == "See the Entries of your data Set":
        st.write("Tabular Format")
        st.write(data.head())

    elif nav == "Enter details of a person":
        st.write("Enter Details of a Person")

        # Create input fields for user to enter details
        person_id = st.text_input("Person ID")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=150, step=1)
        occupation = st.text_input("Occupation")
        sleep_duration = st.number_input("Sleep Duration", min_value=0.0, max_value=24.0, step=0.1)
        quality_of_sleep = st.number_input("Quality of Sleep", min_value=0, max_value=10, step=1)
        physical_activity_level = st.number_input("Physical Activity Level", min_value=0, max_value=100, step=1)
        stress_level = st.number_input("Stress Level", min_value=0, max_value=10, step=1)
        bmi_category = st.selectbox("BMI Category", ["Underweight", "Normal", "Overweight", "Obese"])
        blood_pressure = st.text_input("Blood Pressure")
        heart_rate = st.number_input("Heart Rate", min_value=0, max_value=300, step=1)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=100000, step=1)
        sleep_disorder = st.selectbox("Sleep Disorder", ["None", "Yes", "No"])

        # Submit button to process the entered details
        if st.button("Submit"):
            # Display the entered details
            st.write("Entered Details:")
            details = {
                "Person ID": person_id,
                "Gender": gender,
                "Age": age,
                "Occupation": occupation,
                "Sleep Duration": sleep_duration,
                "Quality of Sleep": quality_of_sleep,
                "Physical Activity Level": physical_activity_level,
                "Stress Level": stress_level,
                "BMI Category": bmi_category,
                "Blood Pressure": blood_pressure,
                "Heart Rate": heart_rate,
                "Daily Steps": daily_steps,
                "Sleep Disorder": sleep_disorder
            }
            st.write(details)

            # Save the details to a CSV file
            details_df = pd.DataFrame([details])
            details_df.to_csv("details.csv", index=False)

    elif nav == "Data Analysis":
        st.write("Data Analysis")

        # Load and preprocess the data
        data = pd.read_csv("data//Sleep_Health_Diseases.csv")

        # Display summary statistics
        st.write("Summary Statistics:")
        st.write(data.describe())

        # Visualize distributions
        st.write("Distributions:")
        for column in data.columns:
            if data[column].dtype in ['int64', 'float64']:
                numericvals.append(data[column].unique())
                st.write(data[column])
                plt.figure(figsize=(8, 6))
                sns.histplot(data[column], kde=True)
                plt.title(f"Distribution of {column}")
                plt.xlabel(column)
                plt.ylabel("Frequency")
                st.pyplot(plt)
                
        # Explore relationships between variables
        st.write("Relationships Between Variables:")
        sns.pairplot(data)
        st.pyplot(plt)
        st.write(numericvals)

if __name__ == "__main__":
    main()
