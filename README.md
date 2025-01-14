# face-recogonization
Introduction
Google Teachable Machine is a user-friendly web-based tool that allows you to create machine learning models for image classification tasks without requiring any coding knowledge. This report will guide you through the process of using Teachable Machine to create a face recognition model and implementing it in Python.

Features:
Image Classification: Train a model to recognize and classify different face images.
User-Friendly Interface: No coding knowledge required to train models.
Exportable Models: Models can be exported and used in various applications.

Installation:
1.Visit the Teachable Machine Website: Go to Teachable Machine
![image](https://github.com/user-attachments/assets/92b02f6e-6868-44a4-ae0d-64c7c19f9c8f)

2.Create a New Project: Click on "Get Started" and select "Image Model" under the "New Project" section
3.Select Model Type: Choose the "Standard Image Model" option.
![image](https://github.com/user-attachments/assets/47881680-2005-44f8-842a-6d80a38ad701)

4.Collect Training Data: Use your webcam or upload images to provide examples for each class (e.g., different faces).
![image](https://github.com/user-attachments/assets/6684bf8b-ff8f-48ab-811d-783182417157)

5.Label Examples: Assign labels to each example
![image](https://github.com/user-attachments/assets/28d39d31-c2b8-47f5-827c-c376af67f347)

6.Train the Model: Click on the "Train" button to start training your model.
![image](https://github.com/user-attachments/assets/93804b00-00e6-4ffe-ba31-d122787d9264)

7.Export the Model: Once training is complete, click on "Export the Model" and download the model files (a .zip file containing the model weights (.h5) and labels (.txt) files)
![image](https://github.com/user-attachments/assets/8c3097c4-a0fb-4da3-9dab-e8f616ae7c00)

#Implementation in Python
1.Set Up Your Environment: Ensure you have Python 3.7 or higher installed.
2.Install Required Libraries: Install OpenCV and NumPy using pip:
python
->pip install opencv-python numpy
3.Extract Model Files: Extract the downloaded .h5 and .txt files from the .zip archive and save them in your project directory.
4.Write Python Code: Use the following code to load the model and perform face recognition:
from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

#CODE EXPLANATION
IMPORTS
from keras.models import load_model:
Imports the load_model function from Keras to load the pre-trained model.
import cv2:
Imports the OpenCV library for computer vision tasks.
import numpy as np:
Imports the NumPy library for numerical operations.

CONFIGURATION
np.set_printoptions(suppress=True):
Sets the NumPy print options to suppress scientific notation for clarity when printing.

LOAD MODELS AND LABELS
model = load_model("keras_Model.h5", compile=False):
Loads the pre-trained model from the file keras_Model.h5 without compiling it.
class_names = open("labels.txt", "r").readlines():
Reads the class labels from the file labels.txt into a list.

CAMERA SETUP
camera = cv2.VideoCapture(0):
Opens the default camera (camera index 0) for capturing images.

MAIN LOOPS
while True::
Starts an infinite loop to continuously capture images.
ret, image = camera.read():
Captures an image from the camera.
image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA):
Resizes the image to 224x224 pixels to match the model's input size.
cv2.imshow("Webcam Image", image):
Displays the captured image in a window titled "Webcam Image".

PREPROCESS IMAGE
image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3):
Converts the image to a NumPy array and reshapes it to (1, 224, 224, 3) to match the model's input shape.
image = (image / 127.5) - 1:
Normalizes the image array to a range of [-1, 1]

MAKE PREDICTION
prediction = model.predict(image):
Uses the model to predict the class of the input image.
index = np.argmax(prediction):
Finds the index of the class with the highest confidence score.
class_name = class_names[index]:
Retrieves the class name corresponding to the predicted index.
confidence_score = prediction[0][index]:
Retrieves the confidence score of the predicted class.

DISPLAY RESULTS
print("Class:", class_name[2:], end=""):
Prints the predicted class name.
print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%"):
Prints the confidence score as a percentage.

HANDLE KEYBOARD INPUT
keyboard_input = cv2.waitKey(1):
Waits for keyboard input.
if keyboard_input == 27::
Checks if the 'Esc' key (ASCII code 27) is pressed to break the loop.

RELEASE RESOURCE
camera.release():
Releases the camera resource.
cv2.destroyAllWindows():
Closes all OpenCV windows.

#CONCLUSION
Using Google Teachable Machine, you can easily create a face recognition model and implement it in Python. This approach is beginner-friendly and customizable, making it a great starting point for learning about machine learning and computer vision.
