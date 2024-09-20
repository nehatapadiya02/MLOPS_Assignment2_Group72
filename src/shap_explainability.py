import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sys
import os

# Add the root directory to sys.path to access 'best_model_pipeline'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from best_model_pipeline import exported_pipeline as best_model_pipeline

# Load and prepare the data
data = pd.read_csv('weather_forecast_csv/processed_weatherHistory.csv')
X = data.drop('Temperature (C)', axis=1)
y = data['Temperature (C)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the data
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

# Extract the model from the pipeline
model = best_model_pipeline

# Create SHAP explainer using the model
explainer = shap.Explainer(model, X_train_preprocessed)

# Generate SHAP values for the test set
shap_values = explainer(X_test_preprocessed)

# Visualize SHAP values and save the plot to a file
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_preprocessed, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig('shap_summary_plot.png')
plt.close()

print("SHAP summary plot saved to 'shap_summary_plot.png'.")
