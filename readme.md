#  AI-Powered Sign Language Interpreter

This project is a lightweight, real-time Sign Language Interpreter that uses MediaPipe for hand tracking and a Machine Learning model to predict sign gestures. It is built with Python, Flask, OpenCV, and Socket.IO.

---

## 🚀 Features

-  Real-time hand gesture recognition using webcam
-  Trained ML model for alphabet and common sign classification
-  Web-based interface using Flask
-  Modular and customizable codebase
-  Offline 

---

#  System Architecture

User Hand → Webcam → MediaPipe (Hand Detection) → Landmark Extraction → Trained ML Model → Prediction → Flask Web App


---

## 🛠️ Installation

```bash
# Clone the repo
git clone https://github.com/AntoVs/SignLanguageInterpreter.git
cd SignLanguageInterpreter

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

# Running the App

```bash
# Using the prepared virtual environment
.\venv\Scripts\python.exe app.py

# Or activate venv and run
venv\Scripts\activate
python app.py
```
*Note: Recommended Python version is Python 3.11 (64-bit).*

# License

This project user lisenced ai model under MIT lisence

# DEMO

![hello](https://github.com/user-attachments/assets/cb88b49a-3853-4699-899f-cc9eb737a212)

![ThankYou](https://github.com/user-attachments/assets/1b327703-3547-4880-b6d7-0d6c283ea758)

