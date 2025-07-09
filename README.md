# Cognifit:- Mental Health Prediction App

**Cognifit** is an AI-driven mental health risk assessment platform, built with Python and Streamlit. It predicts an individual's likelihood of mental health conditions such as **stress, anxiety, and depression** based on lifestyle, behavioral, and demographic data.

The platform is designed for **early screening and awareness**, empowering **students**, **professionals**, and **healthcare institutions** to track, assess, and visualize mental well-being through a user-friendly and interactive dashboard.

---

## Problem Statement

> Mental health issues are on the rise globally, and early detection is critical. Traditional clinical assessments often require face-to-face consultations, which may not be accessible to everyone. **Cognifit** addresses this gap by providing an AI-powered, easily accessible, and data-driven solution that predicts mental health risk levels using structured survey inputs and machine learning algorithms.

---

## Key Features

1. **Mental Health Risk Prediction**
   - Classifies mental health risk into **Low**, **Medium**, or **High**
   - Utilizes **Random Forest** (Scikit-learn) and **Neural Network** (TensorFlow/Keras)

2. **Interactive Assessment Questionnaire**
   - Collects data on:
     - Sleep habits
     - Workload & stress
     - Mood changes
     - Social support
     - Previous mental health conditions

3. **Neural Network-Based Deep Prediction**
   - Implements a **custom Feedforward Neural Network**
   - Captures complex behavioral and emotional patterns

4. **Visual Performance Evaluation**
   - Confusion Matrix
   - ROC-AUC Curve
   - Training Accuracy and Loss graphs

5. **Daily Dashboard & Journaling**
   - Track assessment streaks
   - Monitor risk levels over time
   - Secure journal section for self-reflection

6. **Resource Recommendations**
   - Dynamic links to online mental health support based on your risk level

---

## Directory Structure

```
mental-health-prediction/
├── app.py                  # Streamlit app logic
├── model.py                # ML + DL training scripts
├── mental_health_dataset.csv # Training dataset
├── requirements.txt        # Python dependencies
├── models/
│   ├── rf_model.pkl        # Trained Random Forest
│   ├── nn_model.h5         # Trained Neural Network
│   ├── scaler.pkl          # StandardScaler object
│   └── label_encoder.pkl   # LabelEncoder dictionary
├── history.json            # (optional) Assessment data
├── journal.json            # (optional) User journals
```

---

## How It Works

1. **User fills out the daily assessment form**
2. **ML/DL models process the input**
3. **Risk level is predicted (Low/Medium/High)**
4. **Confidence scores and risk visualizations are displayed**
5. **Streaks, trends, and journals are tracked locally**

---

## Installation & Usage

### Clone the repo:

```bash
git clone https://github.com/your-username/mental-health-prediction.git
cd mental-health-prediction
```

### Run the app:

```bash
streamlit run app.py
```

> Ensure the following models are present in the `models/` directory:
> - `rf_model.pkl`
> - `nn_model.h5`
> - `scaler.pkl`
> - `label_encoder.pkl`

---

## Dataset

The training dataset includes:
- **Categorical Features**: Gender, Employment Status, Work Environment, Mental Health History, Seeks Treatment
- **Numerical Features**: Age, Sleep Hours, Stress Level, Depression/Anxiety Scores, Productivity, Physical Activity, Social Support
- **Label**: Mental Health Risk (Low / Medium / High)

EDA, preprocessing, and model training are done using `model.py`.

---

## Disclaimer

> This app is meant for **self-awareness and educational purposes only**. It is **not a diagnostic tool**. For clinical evaluation, please consult a licensed mental health professional.

---

## Author

**Karan Kumawat**  
Btech CSE(AIML)
SPSU Udaipur
LinkedIn: [karan-kumawat](https://www.linkedin.com/in/karan-kumawat-066b9324b/)

## Note 
The project is build with my team mental health predictor.

---
