import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from lime.lime_tabular import LimeTabularExplainer
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
# Fit and transform the training data, and only transform the test data
X_train_preprocessed = pipeline.fit_transform(X_train)
X_test_preprocessed = pipeline.transform(X_test)

# Create LIME explainer
explainer = LimeTabularExplainer(
    X_train_preprocessed,
    mode='regression',
    feature_names=X.columns,
    training_labels=y_train,
    verbose=False,
    random_state=42
)

# Choose an instance to explain
i = 0
exp = explainer.explain_instance(X_test_preprocessed[i], best_model_pipeline.predict)

# Save LIME explanation to an HTML file
exp.save_to_file('lime_explanation.html')

print("LIME explanation saved to 'lime_explanation.html'.")
