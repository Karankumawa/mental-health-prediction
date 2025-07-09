import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import tensorflow as tf
from tensorflow import keras

# Page configuration
st.set_page_config(
    page_title="MindFlow - Mental Health Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load trained models and preprocessors
@st.cache_resource
def load_models():
    try:
        # Load Random Forest model
        rf_model = joblib.load('models/rf_model.pkl')
        
        # Load scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Load label encoders (should be a dictionary)
        label_encoders = joblib.load('models/label_encoder.pkl')
        
        # Load Neural Network model
        nn_model = keras.models.load_model('models/nn_model.h5')
        
        return rf_model, scaler, label_encoders, nn_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.error("Please ensure all model files are in the 'models/' directory:")
        st.error("- models/rf_model.pkl")
        st.error("- models/scaler.pkl") 
        st.error("- models/label_encoder.pkl")
        st.error("- models/nn_model.h5")
        return None, None, None, None

# Initialize session state
if 'assessments' not in st.session_state:
    st.session_state.assessments = []
if 'journals' not in st.session_state:
    st.session_state.journals = []
if 'streak' not in st.session_state:
    st.session_state.streak = 0

# Load data from files if they exist
def load_data():
    try:
        if os.path.exists('history.json'):
            with open('history.json', 'r') as f:
                st.session_state.assessments = json.load(f)
        if os.path.exists('journal.json'):
            with open('journal.json', 'r') as f:
                st.session_state.journals = json.load(f)
    except:
        pass

# Save data to files
def save_data():
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert assessments and journals to JSON-serializable format
    assessments_serializable = convert_numpy_types(st.session_state.assessments)
    journals_serializable = convert_numpy_types(st.session_state.journals)
    
    with open('history.json', 'w') as f:
        json.dump(assessments_serializable, f)
    with open('journal.json', 'w') as f:
        json.dump(journals_serializable, f)

# Mental Health Prediction Function using trained models
def predict_mental_health(data, rf_model, scaler, label_encoders):
    try:
        # Create DataFrame with the input data
        input_data = pd.DataFrame([data])
        
        # Define categorical and numerical columns (matching your training data)
        categorical_cols = ['gender', 'employment_status', 'work_environment', 'mental_health_history', 'seeks_treatment']
        numerical_cols = ['age', 'stress_level', 'sleep_hours', 'physical_activity_days', 
                         'depression_score', 'anxiety_score', 'social_support_score', 'productivity_score']
        
        # Define the expected feature names based on your training data
        # These should match exactly what was used during training
        expected_features = [
            'age', 'stress_level', 'sleep_hours', 'physical_activity_days',
            'depression_score', 'anxiety_score', 'social_support_score', 'productivity_score',
            'gender_male', 'gender_non-binary',
            'employment_status_Self-employed', 'employment_status_Student', 'employment_status_Unemployed',
            'work_environment_On-site', 'work_environment_Remote',
            'mental_health_history_Yes', 'seeks_treatment_Yes'
        ]
        
        # Create a dataframe with all expected features initialized to 0
        final_input = pd.DataFrame(0, index=[0], columns=expected_features)
        
        # Fill in the numerical features
        for col in numerical_cols:
            if col in input_data.columns:
                final_input[col] = input_data[col].iloc[0]
        
        # Fill in the categorical features based on input
        # Gender
        if data.get('gender') == 'male':
            final_input['gender_male'] = 1
        elif data.get('gender') == 'non-binary':
            final_input['gender_non-binary'] = 1
        # If female, both remain 0 (reference category)
        
        # Employment Status
        emp_status = data.get('employment_status', '').lower()
        if emp_status == 'self-employed':
            final_input['employment_status_Self-employed'] = 1
        elif emp_status == 'student':
            final_input['employment_status_Student'] = 1
        elif emp_status == 'unemployed':
            final_input['employment_status_Unemployed'] = 1
        # If employed, all remain 0 (reference category)
        
        # Work Environment
        work_env = data.get('work_environment', '').lower()
        if work_env == 'on-site':
            final_input['work_environment_On-site'] = 1
        elif work_env == 'remote':
            final_input['work_environment_Remote'] = 1
        # If hybrid, both remain 0 (reference category)
        
        # Mental Health History
        if data.get('mental_health_history') == 'yes':
            final_input['mental_health_history_Yes'] = 1
        
        # Seeks Treatment
        if data.get('seeks_treatment') == 'yes':
            final_input['seeks_treatment_Yes'] = 1
        
        # Scale numerical features
        final_input[numerical_cols] = scaler.transform(final_input[numerical_cols])
        
        # Make prediction using Random Forest
        prediction_proba = rf_model.predict_proba(final_input)[0]
        prediction_class = rf_model.predict(final_input)[0]
        
        # Map prediction to risk levels
        risk_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
        prediction_label = risk_mapping.get(prediction_class, 'Unknown')
        
        # Create confidence dictionary with Python native types
        confidence = {
            'low': float(prediction_proba[0]) if len(prediction_proba) > 0 else 0.33,
            'medium': float(prediction_proba[1]) if len(prediction_proba) > 1 else 0.33,
            'high': float(prediction_proba[2]) if len(prediction_proba) > 2 else 0.33
        }
        
        # Smart override for high-risk red flags
        if data.get('stress_level', 5) >= 8 and data.get('sleep_hours', 7) < 4:
            return {
                'prediction': 'High',
                'confidence': {'low': 0.1, 'medium': 0.2, 'high': 0.9},
                'risk_level': 3
            }
        
        return {
            'prediction': prediction_label,
            'confidence': confidence,
            'risk_level': int(prediction_class) + 1  # Convert to Python int and add 1
        }
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Fallback to simple rule-based prediction
        return fallback_prediction(data)

# Fallback prediction function (in case model loading fails)
def fallback_prediction(data):
    stress_level = data.get('stress_level', 5)
    sleep_hours = data.get('sleep_hours', 7)
    depression_score = data.get('depression_score', 10)
    anxiety_score = data.get('anxiety_score', 5)
    
    # Smart override for high-risk red flags
    if stress_level >= 8 and sleep_hours < 4:
        return {
            'prediction': 'High',
            'confidence': {'low': 0.1, 'medium': 0.2, 'high': 0.9},
            'risk_level': 3
        }
    
    # Simple risk calculation based on multiple factors
    risk_score = (stress_level * 0.3) + ((10 - sleep_hours) * 0.2) + (depression_score * 0.25) + (anxiety_score * 0.25)
    
    if risk_score <= 3:
        return {
            'prediction': 'Low',
            'confidence': {'low': 0.8, 'medium': 0.15, 'high': 0.05},
            'risk_level': 1
        }
    elif risk_score <= 6:
        return {
            'prediction': 'Medium',
            'confidence': {'low': 0.2, 'medium': 0.7, 'high': 0.1},
            'risk_level': 2
        }
    else:
        return {
            'prediction': 'High',
            'confidence': {'low': 0.1, 'medium': 0.3, 'high': 0.6},
            'risk_level': 3
        }

# Get mental health resources based on risk level
def get_resources(risk_level):
    resources = {
        'Low': [
            {"title": "Mindfulness Meditation", "url": "https://www.headspace.com"},
            {"title": "Mental Health Tips", "url": "https://www.mentalhealth.gov/basics/what-is-mental-health"}
        ],
        'Medium': [
            {"title": "Crisis Text Line", "url": "https://www.crisistextline.org"},
            {"title": "Psychology Today", "url": "https://www.psychologytoday.com"},
            {"title": "NAMI Support", "url": "https://www.nami.org"}
        ],
        'High': [
            {"title": "National Suicide Prevention Lifeline", "url": "tel:988"},
            {"title": "Crisis Text Line", "url": "https://www.crisistextline.org"},
            {"title": "Emergency Services", "url": "tel:911"},
            {"title": "SAMHSA Helpline", "url": "tel:1-800-662-4357"}
        ]
    }
    return resources.get(risk_level, resources['Low'])

# Calculate streak
def calculate_streak():
    if not st.session_state.assessments:
        return 0
    
    dates = [datetime.strptime(a['date'], '%Y-%m-%d') for a in st.session_state.assessments]
    dates.sort(reverse=True)
    
    if not dates:
        return 0
    
    streak = 1
    for i in range(1, len(dates)):
        if (dates[i-1] - dates[i]).days == 1:
            streak += 1
        else:
            break
    
    return streak

# Load models at startup
rf_model, scaler, label_encoders, nn_model = load_models()

# Load data on startup
load_data()

# Check if models loaded successfully
if rf_model is None:
    st.error("⚠️ **Models not loaded!** The app will use a fallback prediction system.")
    st.info("To use the trained ML models, please ensure all model files are in the 'models/' directory.")

# Sidebar Navigation
st.sidebar.title("🧠 MindFlow")
page = st.sidebar.selectbox(
    "Navigate",
    ["🏠 Dashboard", "📋 Assessment", "📈 Tracking", "📝 Journal", "📚 Resources"]
)

# Main App Logic
if page == "🏠 Dashboard":
    st.markdown('<h1 class="main-header">🧠 MindFlow Dashboard</h1>', unsafe_allow_html=True)
    
    # Model status indicator
    if rf_model is not None:
        st.success("✅ **AI Models Loaded Successfully** - Using trained Random Forest and Neural Network models")
    else:
        st.warning("⚠️ **Using Fallback System** - Trained models not available")
    
    # Welcome message
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;">
        <h2>Welcome back to your mental wellness journey! 🌟</h2>
        <p>Track your mental health with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Calculate current streak
    current_streak = calculate_streak()
    st.session_state.streak = current_streak
    
    # Get recent assessment
    recent_assessment = st.session_state.assessments[-1] if st.session_state.assessments else None
    
    # Stats Grid
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Check-in Streak", f"{current_streak} days", "🔥")
    
    with col2:
        if recent_assessment:
            risk_color = "🟢" if recent_assessment['prediction'] == 'Low' else "🟡" if recent_assessment['prediction'] == 'Medium' else "🔴"
            st.metric("Current Risk", f"{recent_assessment['prediction']} {risk_color}")
        else:
            st.metric("Current Risk", "No data")
    
    with col3:
        sleep_goal = 7
        if recent_assessment:
            sleep_status = "✅ Achieved" if recent_assessment['sleep_hours'] >= sleep_goal else "❌ Missed"
            st.metric("Sleep Goal", sleep_status)
        else:
            st.metric("Sleep Goal", "No data")
    
    with col4:
        st.metric("Total Entries", len(st.session_state.assessments), "📊")
    
    # Model Confidence Display
    if recent_assessment:
        st.subheader("🔮 AI Model Confidence")
        confidence_data = recent_assessment['confidence']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Low Risk", f"{confidence_data['low']*100:.1f}%")
        with col2:
            st.metric("Medium Risk", f"{confidence_data['medium']*100:.1f}%")
        with col3:
            st.metric("High Risk", f"{confidence_data['high']*100:.1f}%")
        
        # Confidence bar chart
        fig = go.Figure(data=[
            go.Bar(name='Confidence', 
                   x=['Low Risk', 'Medium Risk', 'High Risk'],
                   y=[confidence_data['low']*100, confidence_data['medium']*100, confidence_data['high']*100],
                   marker_color=['#4CAF50', '#FF9800', '#f44336'])
        ])
        fig.update_layout(title="Model Confidence Levels", yaxis_title="Confidence (%)")
        st.plotly_chart(fig, use_container_width=True)
    
    # Instant Interventions
    if recent_assessment and recent_assessment['prediction'] in ['Medium', 'High']:
        st.warning("🚨 **Wellness Check**: Your recent assessment indicates elevated stress. Consider taking a few minutes for self-care.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Write in Journal"):
                st.session_state.page = "📝 Journal"
                st.rerun()
        with col2:
            if st.button("🧘 Breathing Exercise"):
                st.info("Take 5 deep breaths: Inhale for 4 counts, hold for 4, exhale for 6. Repeat 5 times.")
    
    # Quick Actions
    st.subheader("Quick Actions")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Daily Check-in", use_container_width=True):
            st.session_state.page = "📋 Assessment"
            st.rerun()
    
    with col2:
        if st.button("📊 Export Insights", use_container_width=True) and recent_assessment:
            insights = {
                'date': recent_assessment['date'],
                'mood': recent_assessment['prediction'],
                'sleep': recent_assessment['sleep_hours'],
                'stress': recent_assessment['stress_level'],
                'streak': current_streak
            }
            st.download_button(
                label="Download JSON",
                data=json.dumps(insights, indent=2),
                file_name=f"mental-health-insights-{insights['date']}.json",
                mime="application/json"
            )

elif page == "📋 Assessment":
    st.markdown('<h1 class="main-header">📋 Daily Mental Health Assessment</h1>', unsafe_allow_html=True)
    
    with st.form("assessment_form"):
        st.subheader("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=25)
        
        with col2:
            gender = st.selectbox("Gender", ["male", "female", "non-binary", "prefer not to say"])
        
        # Additional categorical fields from your model
        col1, col2 = st.columns(2)
        with col1:
            employment_status = st.selectbox("Employment Status", 
                                           ["employed", "self-employed", "student", "unemployed"])
        
        with col2:
            work_environment = st.selectbox("Work Environment", 
                                          ["on-site", "remote", "hybrid"])
        
        col1, col2 = st.columns(2)
        with col1:
            mental_health_history = st.selectbox("Mental Health History", ["no", "yes"])
        
        with col2:
            seeks_treatment = st.selectbox("Currently Seeking Treatment", ["no", "yes"])
        
        st.subheader("Health Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        
        with col2:
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        
        col1, col2 = st.columns(2)
        with col1:
            depression_score = st.slider("Depression Score (0-30)", 0, 30, 0)
        
        with col2:
            anxiety_score = st.slider("Anxiety Score (0-21)", 0, 21, 0)
        
        col1, col2 = st.columns(2)
        with col1:
            social_support_score = st.slider("Social Support (0-100)", 0, 100, 50)
        
        with col2:
            physical_activity_days = st.number_input("Physical Activity (days/week)", 0, 7, 3)
        
        productivity_score = st.slider("Productivity Score (0-100)", 0, 100, 50)
        
        st.subheader("Journal Entry (Optional)")
        journal_entry = st.text_area("How are you feeling today?", height=100)
        
        submitted = st.form_submit_button("Complete Assessment", use_container_width=True)
        
        if submitted:
            # Create assessment data
            assessment_data = {
                'age': age,
                'gender': gender,
                'employment_status': employment_status,
                'work_environment': work_environment,
                'mental_health_history': mental_health_history,
                'seeks_treatment': seeks_treatment,
                'sleep_hours': sleep_hours,
                'stress_level': stress_level,
                'depression_score': depression_score,
                'anxiety_score': anxiety_score,
                'social_support_score': social_support_score,
                'physical_activity_days': physical_activity_days,
                'productivity_score': productivity_score,
                'journal_entry': journal_entry
            }
            
            # Get prediction using trained model or fallback
            if rf_model is not None:
                prediction = predict_mental_health(assessment_data, rf_model, scaler, label_encoders)
            else:
                prediction = fallback_prediction(assessment_data)
            
            # Create full assessment record
            assessment = {
                'id': len(st.session_state.assessments) + 1,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timestamp': datetime.now().isoformat(),
                **assessment_data,
                **prediction
            }
            
            # Add to session state
            st.session_state.assessments.append(assessment)
            
            # Add journal entry if provided
            if journal_entry:
                journal_record = {
                    'id': len(st.session_state.journals) + 1,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'content': journal_entry,
                    'mood': prediction['prediction']
                }
                st.session_state.journals.append(journal_record)
            
            # Save data
            save_data()
            
            # Show results
            st.success("Assessment completed successfully!")
            
            # Display prediction
            risk_class = f"risk-{prediction['prediction'].lower()}"
            st.markdown(f"""
            <div class="{risk_class}">
                <h3>Mental Health Risk: {prediction['prediction']}</h3>
                <p>Risk Level: {prediction['risk_level']}/3</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence
            st.subheader("Model Confidence")
            conf_col1, conf_col2, conf_col3 = st.columns(3)
            with conf_col1:
                st.metric("Low", f"{prediction['confidence']['low']*100:.1f}%")
            with conf_col2:
                st.metric("Medium", f"{prediction['confidence']['medium']*100:.1f}%")
            with conf_col3:
                st.metric("High", f"{prediction['confidence']['high']*100:.1f}%")

elif page == "📈 Tracking":
    st.markdown('<h1 class="main-header">📈 Mood & Sleep Tracking</h1>', unsafe_allow_html=True)
    
    if not st.session_state.assessments:
        st.info("No data available yet. Complete your first assessment to see trends!")
    else:
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(st.session_state.assessments)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Last 7 days data
        last_7_days = df.tail(7)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sleep Hours Trend', 'Stress Level Trend', 
                          'Risk Level Trend', 'Goal Achievement'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sleep trend
        fig.add_trace(
            go.Scatter(x=last_7_days['date'], y=last_7_days['sleep_hours'],
                      mode='lines+markers', name='Sleep Hours',
                      line=dict(color='#1f77b4', width=3)),
            row=1, col=1
        )
        
        # Stress trend
        fig.add_trace(
            go.Scatter(x=last_7_days['date'], y=last_7_days['stress_level'],
                      mode='lines+markers', name='Stress Level',
                      line=dict(color='#ff7f0e', width=3)),
            row=1, col=2
        )
        
        # Risk level trend
        fig.add_trace(
            go.Scatter(x=last_7_days['date'], y=last_7_days['risk_level'],
                      mode='lines+markers', name='Risk Level',
                      line=dict(color='#d62728', width=3)),
            row=2, col=1
        )
        
        # Goal achievement
        sleep_goal = 7
        goal_achievement = [1 if hours >= sleep_goal else 0 for hours in last_7_days['sleep_hours']]
        fig.add_trace(
            go.Bar(x=last_7_days['date'], y=goal_achievement,
                   name='Goal Achievement', marker_color='#2ca02c'),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Mental Health Trends (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_sleep = last_7_days['sleep_hours'].mean()
            st.metric("Avg Sleep", f"{avg_sleep:.1f}h")
        
        with col2:
            avg_stress = last_7_days['stress_level'].mean()
            st.metric("Avg Stress", f"{avg_stress:.1f}/10")
        
        with col3:
            goals_met = sum(goal_achievement)
            st.metric("Goals Met", f"{goals_met}/7 days")
        
        with col4:
            risk_distribution = last_7_days['prediction'].value_counts()
            most_common_risk = risk_distribution.index[0] if len(risk_distribution) > 0 else "N/A"
            st.metric("Most Common Risk", most_common_risk)

elif page == "📝 Journal":
    st.markdown('<h1 class="main-header">📝 Secure Journal</h1>', unsafe_allow_html=True)
    
    # Add new journal entry
    st.subheader("✍️ New Entry")
    with st.form("journal_form"):
        journal_content = st.text_area("What's on your mind today?", height=150)
        mood_rating = st.selectbox("How are you feeling?", ["Great", "Good", "Okay", "Not Good", "Terrible"])
        
        if st.form_submit_button("Save Entry"):
            if journal_content:
                journal_entry = {
                    'id': len(st.session_state.journals) + 1,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'timestamp': datetime.now().isoformat(),
                    'content': journal_content,
                    'mood': mood_rating
                }
                st.session_state.journals.append(journal_entry)
                save_data()
                st.success("Journal entry saved!")
            else:
                st.error("Please write something before saving.")
    
    # Display journal entries
    st.subheader("📖 Your Entries")
    if not st.session_state.journals:
        st.info("No journal entries yet. Write your first entry above!")
    else:
        # Sort by date (newest first)
        sorted_journals = sorted(st.session_state.journals, key=lambda x: x['date'], reverse=True)
        
        for entry in sorted_journals:
            with st.expander(f"📅 {entry['date']} - Mood: {entry['mood']}"):
                st.write(entry['content'])
                if 'timestamp' in entry:
                    st.caption(f"Written at: {entry['timestamp']}")

elif page == "📚 Resources":
    st.markdown('<h1 class="main-header">📚 Mental Health Resources</h1>', unsafe_allow_html=True)
    
    # Get current risk level
    recent_assessment = st.session_state.assessments[-1] if st.session_state.assessments else None
    risk_level = recent_assessment['prediction'] if recent_assessment else 'Low'
    
    # Risk level indicator
    risk_colors = {'Low': '#4CAF50', 'Medium': '#FF9800', 'High': '#f44336'}
    st.markdown(f"""
    <div style="background: {risk_colors[risk_level]}; color: white; padding: 1rem; 
                border-radius: 10px; text-align: center; margin-bottom: 2rem;">
        <h3>Current Risk Level: {risk_level}</h3>
        <p>Resources tailored to your current mental health status</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get appropriate resources
    resources = get_resources(risk_level)
    
    st.subheader(f"🎯 Resources for {risk_level} Risk Level")
    
    # Display resources in cards
    for i, resource in enumerate(resources):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{resource['title']}**")
        with col2:
            st.link_button("Access", resource['url'])
    
    # Emergency contacts
    st.subheader("🚨 Emergency Contacts")
    emergency_contacts = [
        {"service": "National Suicide Prevention Lifeline", "number": "988"},
        {"service": "Crisis Text Line", "number": "Text HOME to 741741"},
        {"service": "SAMHSA Helpline", "number": "1-800-662-4357"},
        {"service": "Emergency Services", "number": "911"}
    ]
    
    for contact in emergency_contacts:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{contact['service']}**")
        with col2:
            st.markdown(f"`{contact['number']}`")
    
    # Self-care tips
    st.subheader("💡 Self-Care Tips")
    tips = [
        "🧘 Practice mindfulness meditation for 10 minutes daily",
        "🚶 Take a 15-minute walk in nature",
        "📱 Limit social media usage before bedtime",
        "💤 Maintain a consistent sleep schedule",
        "🤝 Connect with friends and family regularly",
        "📝 Write in a gratitude journal",
        "🎵 Listen to calming music",
        "🫁 Practice deep breathing exercises"
    ]
    
    for tip in tips:
        st.markdown(f"• {tip}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🧠 MindFlow - Your AI-Powered Mental Health Companion</p>
    <p><small>Remember: This app is for tracking purposes only. Please consult healthcare professionals for medical advice.</small></p>
</div>
""", unsafe_allow_html=True)