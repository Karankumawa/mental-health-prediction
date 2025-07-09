# Create a app for the mental Health Prediction by using the supervised learning (Random Forest classifier) and the Neural Networking

# Load the libaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder


#models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("mental_health_dataset.csv")

print("Initial Shape:", df.shape)
df.head()

# Processing the EDA on the Dataset
# 1. NULL VALUE CHECK
df.isna().sum(axis=0)

# Drop rows with any nulls (or you can use imputation)
df.dropna(inplace=True)
print("After Dropping Nulls:", df.shape)

# 2. CLEAN GENDER COLUMN
df["gender"].unique()
df["gender"].value_counts()

# Normalize gender values
df['gender'] = df['gender'].str.lower().str.strip()
df['gender'] = df['gender'].replace({
    'm': 'male', 'mail': 'male',
    'f': 'female', 'femlae': 'female', 'femal': 'female',
    'non binary': 'non-binary', 'trans': 'non-binary'
})

# Filter valid gender values only
valid_genders = ['male', 'female', 'non-binary']
df = df[df['gender'].isin(valid_genders)]
print(" Cleaned Gender Values:", df['gender'].unique())

# 3. VALIDATE AGE RANGE
print("Age Summary:\n", df['age'].describe())

# Remove unrealistic ages
df = df[(df['age'] >= 10) & (df['age'] <= 100)]

# 4. OUTLIER DETECTION & REMOVAL
def remove_outliers_iqr(dataframe, col):
    Q1 = dataframe[col].quantile(0.25)
    Q3 = dataframe[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return dataframe[(dataframe[col] >= lower) & (dataframe[col] <= upper)]

# Apply to all numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in num_cols:
    df = remove_outliers_iqr(df, col)

print("Final Cleaned Data Summary:\n", df.describe())
print("Final Shape after EDA:", df.shape)

# Pre-Processing of Dataset

# 1. Separate Features & Target
target_col = 'mental_health_risk'
X = df.drop(columns=[target_col])
y = df[target_col]


# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 2. Encode Target Labels (Low, Medium, High → 0, 1, 2)
# Assuming you have your preprocessed DataFrame 'df' or the individual series of categorical data
# Example data (replace with your actual data loading and preprocessing steps)
data = {
    'gender': ['Male', 'Female', 'Non-binary', 'Male', 'Female'],
    'employment_status': ['Employed', 'Student', 'Employed', 'Unemployed', 'Student'],
    'work_environment': ['On-site', 'Remote', 'Hybrid', 'On-site', 'Remote'],
    'mental_health_history': ['No', 'Yes', 'No', 'Yes', 'No'],
    'seeks_treatment': ['No', 'Yes', 'No', 'No', 'Yes'],
    'mental_health_risk': ['Low', 'Medium', 'Low', 'High', 'Medium'] # Your target variable
}
df = pd.DataFrame(data)

# Dictionary to hold all label encoders
label_encoders = {}

# List of categorical features from your dataset that need encoding
categorical_features_for_input = [
    'gender', 'employment_status', 'work_environment',
    'mental_health_history', 'seeks_treatment'
]
target_feature = 'mental_health_risk' # Your target variable

# Fit and store LabelEncoders for input features
for feature in categorical_features_for_input:
    le = LabelEncoder()
    # Ensure you fit on the entire column of your training data
    label_encoders[feature] = le.fit(df[feature]) # Use your actual training data's column

# Fit and store LabelEncoder for the target variable (mental_health_risk)
le_risk = LabelEncoder()
label_encoders[target_feature] = le_risk.fit(df[target_feature]) # Use your actual training data's target column

# Create the models directory if it doesn't exist
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)

# Save the dictionary of label encoders
joblib.dump(label_encoders, os.path.join(models_dir, 'label_encoder.pkl'))

print("All label encoders saved successfully as a dictionary in models/label_encoder.pkl")


# 3. Identify Categorical and Numerical Columns
categorical_cols = ['gender', 'employment_status', 'work_environment', 'mental_health_history', 'seeks_treatment']
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols = [col for col in numerical_cols if col not in categorical_cols]

# 4. One-Hot Encode Categorical Features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# 5. Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

joblib.dump(scaler, "models/scaler.pkl")

X_encoded.head()

# 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

#................................. RANDOM FOREST MODEL........................................

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, "models/rf_model.pkl")
y_pred_rf = rf_model.predict(X_test)
print(" Random Forest Report:\n", classification_report(y_test, y_pred_rf))

#---------------------------------NEURAL NETWORK MODEL-------------------------------------------

print("Training Neural Network...")
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')
])

#Compile the model
nn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',           # Monitor validation loss
    patience=5,                   # Wait 5 epochs without improvement before stopping
    restore_best_weights=True    # Restore the best weights after stopping
)

history = nn_model.fit(X_train, y_train_cat,validation_split=0.2,epochs=30,batch_size=32,callbacks=[early_stop],verbose=1)

nn_model.save("models/nn_model.h5")

nn_model.evaluate(X_test, y_test_cat)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Value")
plt.legend()
plt.show()