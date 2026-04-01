# Word Sense Disambiguation System (WSD)

## Overview
This project implements a Word Sense Disambiguation (WSD) system using a hybrid approach combining Machine Learning and rule-based techniques.

The system predicts the correct meaning of ambiguous words (such as "bank" or "bat") based on the context of a sentence.

---

## Features
- Machine Learning model using Naive Bayes  
- TF-IDF based feature extraction  
- Rule-based refinement for improved accuracy  
- Streamlit web interface  
- Excel dataset testing support  
- Real-time prediction with confidence scores  

---

## Methodology
1. Dataset is preprocessed and labeled using rule-based logic  
2. TF-IDF is used to convert text into numerical features  
3. A Naive Bayes classifier is trained on the dataset  
4. A hybrid approach is used:
   - ML model for general prediction  
   - Rule-based override for contextual accuracy  

---

## Model Performance
- Accuracy: ~85%  
- Evaluation Method: Train-Test Split (80-20)  

---

## Project Structure

app.py               - Streamlit application  
model.pkl            - Trained ML model  
vectorizer.pkl       - TF-IDF vectorizer  
requirements.txt     - Dependencies  
README.md            - Documentation  

---

## Installation

### Step 1: Clone the repository
git clone https://github.com/YOUR_USERNAME/wsd-ml-project.git  
cd wsd-ml-project  

### Step 2: Create virtual environment
python3 -m venv .venv  
source .venv/bin/activate  

### Step 3: Install dependencies
pip install -r requirements.txt  

---

## Run the Application
streamlit run app.py  

---

## Deployment
The application can be deployed using Streamlit Cloud by connecting the GitHub repository and selecting the app.py file.

---

## Author
Name: Mohan Narayanapuram  
Register Number: RA2311056010126  
Course: 21CSE356T - Natural Language Processing  
Institution: SRM Institute of Science and Technology  