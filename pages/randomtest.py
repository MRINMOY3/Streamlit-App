import streamlit as st
import cv2
import face_recognition
import os
import csv
import subprocess

# Function to check if the Streamlit app is already running
def is_streamlit_running():
    with os.popen("tasklist /FI \"IMAGENAME eq streamlit.exe\"") as tasks:
        for task in tasks:
            if "streamlit.exe" in task:
                return True
    return False

def register_face():
    name = st.text_input("Enter the name of the person:")
    st.write("Press 'Capture' to register the face.")
    
    # Placeholder for displaying webcam feed
    camera_output = st.empty()
    
    # Initialize Streamlit Camera Input
    capture_button = st.button("Capture")
    
    if capture_button:
        # Initialize webcam
        video_capture = cv2.VideoCapture(0)
        
        # Capture frame-by-frame from webcam
        ret, frame = video_capture.read()
        
        # Display the webcam feed
        camera_output.image(frame, channels='BGR', use_column_width=True)
        
        # Save captured image
        image_path = f"faces/{name}.jpg"
        cv2.imwrite(image_path, frame)
        st.write(f"Face registered successfully as {name}")
            
        # Save face data to a CSV file
        with open('registered_faces.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([name, image_path])
        
        # Release the webcam
        video_capture.release()

def login():
    st.write("Validating in...")
    # Load registered face data from the CSV file
    registered_faces = []
    with open('registered_faces.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            registered_faces.append(row)

    # Streamlit placeholder for login output
    login_output = st.empty()

    # Placeholder for displaying webcam feed
    camera_output = st.empty()

    # Check if Streamlit app is already running
    if not is_streamlit_running():
        # Streamlit Camera Input
        camera = cv2.VideoCapture(0)

        # Flag to check if redirection has been done
        redirected = False

        while True:
            ret, frame = camera.read()
            camera_output.image(frame, channels='BGR', use_column_width=True)

            # Find face locations and encodings in the webcam frame
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            # Compare webcam face encodings with the registered face encodings
            # Compare webcam face encodings with the registered face encodings
            for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                for registered_face in registered_faces:
                    registered_name, registered_image_path = registered_face
                    # Construct the absolute path of the image file
                    script_directory = os.path.dirname(os.path.abspath(__file__))
                    image_path = os.path.join(script_directory, registered_image_path)
                    registered_image = face_recognition.load_image_file(image_path)
                    
                    # Get face encodings for the registered image
                    registered_face_encodings = face_recognition.face_encodings(registered_image)
                    
                    if len(registered_face_encodings) > 0:
                        registered_face_encoding = registered_face_encodings[0]
                        match = face_recognition.compare_faces([registered_face_encoding], face_encoding)
                        
                        # If match found and redirection hasn't been done yet, redirect to another Python app
                        if match[0] and not redirected:
                            try:
                                parent_directory = os.path.dirname(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
                                f1_path = os.path.join(parent_directory,"\A1\Homepage.py")

                                login_output.write(f"Welcome {registered_name}!\n")
                                subprocess.Popen(["python", "-m", "streamlit", "run", "MAIN_APP.py"])
                                redirected = True
                            except Exception as e:
                                print(f"Error occurred while redirecting: {e}")

                            # Additional logic after redirection
                            if redirected:
                                with open("login_output.txt", "a") as login_output:
                                    login_output.write(f"Welcome {registered_name}!\n")
                                    login_output.write("Now logging you in to the main Contents...\n")

                                    # Break the loop if 'q' is pressed
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        break


        camera.release()
    else:
        st.write("Streamlit app is already running.")

def main():
    st.title("Face Recognition System")
    choice = st.sidebar.selectbox("Menu:", ["Register new face", "Login", "Exit"])

    if choice == "Register new face":
        register_face()
    elif choice == "Login":
        login()
    elif choice == "Exit":
        st.write("Exiting program...")

if __name__ == "__main__":
    main()
