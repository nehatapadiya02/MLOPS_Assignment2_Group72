import pytest
import pandas as pd
from sklearn.model_selection import train_test_split  # Ensure this import is present
from unittest.mock import patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_selection import load_data, train_and_evaluate_model

@pytest.fixture
def sample_processed_data():
    data = {
        'Temperature (C)': [0.5, 0.6],
        'Humidity': [0.8, 0.85],
        'Wind Speed (km/h)': [0.333, 0.4],
        'Visibility (km)': [0.7, 0.7],
        'Pressure (millibars)': [0.6, 0.7],
        'Hour': [0, 1],
        'DayOfWeek': [2, 3],
        'Month': [1, 1],
        'Precip Type_rain': [1, 0],
        'Precip Type_snow': [0, 1],
        'TempDiff': [1.0, 1.0],
        'Summary_encoded': [20.0, 21.0]
    }
    df = pd.DataFrame(data)
    return df

@patch('src.model_selection.TPOTRegressor')
def test_train_and_evaluate_model(mock_tpot, sample_processed_data):
    # Prepare data
    X = sample_processed_data.drop('Temperature (C)', axis=1)
    y = sample_processed_data['Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Mock TPOTRegressor behavior
    mock_model = mock_tpot.return_value
    mock_model.predict.return_value = [0.55] 
    mock_model.fit.return_value = mock_model

    # Call the function
    train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # Assertions
    mock_tpot.assert_called_once_with(verbosity=2, generations=1, population_size=2, random_state=42)
    mock_model.fit.assert_called_once_with(X_train, y_train)
    mock_model.predict.assert_called_once_with(X_test)
