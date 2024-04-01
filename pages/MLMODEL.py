import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Declare global variable for the model
global trained_model
trained_model = None


def train_model(X_train, y_train):
    global trained_model  # Access the global variable
    model = LinearRegression()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    trained_model = model  # Assign the trained model to the global variable
    print("Model trained and stored in global variable.")
    return model

# Function to load and preprocess the data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)  # Drop rows with missing values
    return data

# Function to train the Multiple Linear Regression model
def train_model(X_train, y_train):
    global trained_model  # Access the global variable
    model = LinearRegression()
    model.fit(X_train, y_train)
    trained_model = model  # Assign the trained model to the global variable
    print("Model trained and stored in global variable.")
    return model

# Function to make predictions
# Function to make predictions
def predict(model, user_data, feature_names):
    global trained_model
    if trained_model is not None:
        return trained_model.predict(user_data), feature_names
    else:
        raise ValueError("Model has not been trained yet.")


# Function for user input of features
# def user_input_features():
#     age = st.number_input("Enter age", min_value=0, max_value=150, value=30)
#     sleep_duration = st.number_input("Enter sleep duration (in hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
#     quality_of_sleep = st.slider("Enter quality of sleep (on a scale of 1 to 10)", min_value=1, max_value=10, value=5)
#     physical_activity_level = st.number_input("Enter physical activity level (in minutes per day)", min_value=0, max_value=1440, value=30)
#     heart_rate = st.number_input("Enter heart rate", min_value=0, max_value=300, value=60)
#     daily_steps = st.number_input("Enter daily steps", min_value=0, max_value=50000, value=5000)

#     return np.array([[age, sleep_duration, quality_of_sleep, physical_activity_level, heart_rate, daily_steps]])

# Main function to run the Streamlit app
def main():
    st.sidebar.title("NAVIGATION PANEL")
    nav = st.sidebar.radio("Navigation", ["Home", "Data Analysis and Prediction"])

    if nav == "Home":
        st.write("# Stress Level Prediction")
        st.write("Use the sidebar to navigate")

    elif nav == "Data Analysis and Prediction":
        st.write("# Stress Level Prediction - Data Analysis")

        # Load and preprocess the data
        data = load_and_preprocess_data("data//Sleep_Health_Diseases.csv")

        # Selecting features and target
        X = data[['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']]
        y = data['Stress Level']

        # Splitting the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)  # Convert to arrays of values

        # Train the model and store it in the global variable
        train_model(X_train, y_train)

        # Evaluation metrics
        y_pred = trained_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write("## Model Evaluation")
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared Score: {r2}")

        if trained_model is None:  # Check if model is trained
            st.write("Please train the model in the 'Data Analysis' section first.")
        else:
                feature_names = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Heart Rate', 'Daily Steps']

                age = st.number_input("Enter age", min_value=0, max_value=150, value=30)
                sleep_duration = st.number_input("Enter sleep duration (in hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
                quality_of_sleep = st.slider("Enter quality of sleep (on a scale of 1 to 10)", min_value=1, max_value=10, value=5)
                physical_activity_level = st.number_input("Enter physical activity level (in minutes per day)", min_value=0, max_value=1440, value=30)
                heart_rate = st.number_input("Enter heart rate", min_value=0, max_value=300, value=60)
                daily_steps = st.number_input("Enter daily steps", min_value=0, max_value=50000, value=5000)

                user_data = np.array([[age, sleep_duration, quality_of_sleep, physical_activity_level, heart_rate, daily_steps]])
                predicted_stress_level, _ = predict(trained_model, user_data, feature_names)
                if st.button("Ready To submit the details ?"):
                 st.write("Predicted stress level:", predicted_stress_level)

if __name__ == "__main__":
    main()



## THE FINAL CODE