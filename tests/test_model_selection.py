import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from unittest.mock import patch
from sklearn.metrics import mean_squared_error
# from src.model_selection import train_and_evaluate_model
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model_selection import load_data, train_and_evaluate_model

@pytest.fixture
def sample_processed_data():
    data = {
        'Temperature (C)': [0.5, 0.6, 0.7, 0.8],
        'Humidity': [0.8, 0.85, 0.9, 0.95],
        'Wind Speed (km/h)': [0.333, 0.4, 0.5, 0.6],
        'Visibility (km)': [0.7, 0.7, 0.8, 0.9],
        'Pressure (millibars)': [0.6, 0.7, 0.8, 0.9],
        'Hour': [0, 1, 2, 3],
        'DayOfWeek': [2, 3, 4, 5],
        'Month': [1, 1, 1, 1],
        'Precip Type_rain': [1, 0, 0, 1],
        'Precip Type_snow': [0, 1, 1, 0],
        'TempDiff': [1.0, 1.0, 1.0, 1.0],
        'Summary_encoded': [20.0, 21.0, 22.0, 23.0]
    }
    df = pd.DataFrame(data)
    return df

@patch('src.model_selection.TPOTRegressor')
@patch('src.model_selection.joblib.dump')
@patch('src.model_selection.mean_squared_error')
def test_train_and_evaluate_model(mock_mse, mock_joblib_dump, mock_tpot, sample_processed_data):
    # Prepare data
    X = sample_processed_data.drop('Temperature (C)', axis=1)
    y = sample_processed_data['Temperature (C)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Mock TPOTRegressor behavior
    mock_model = mock_tpot.return_value
    mock_model.predict.return_value = np.array([0.55, 0.65])  # Mock predictions
    mock_model.fit.return_value = mock_model
    mock_model.fitted_pipeline_ = 'mocked_pipeline'  # Mock fitted pipeline

    # Mock mean_squared_error to return a fixed MSE
    mock_mse.return_value = 0.01

    # Call the function
    train_and_evaluate_model(X_train, y_train, X_test, y_test)

    # Assertions
    mock_tpot.assert_called_once_with(verbosity=2, generations=5, population_size=10, random_state=42)
    mock_model.fit.assert_called_once_with(X_train, y_train)
    mock_model.predict.assert_called_once_with(X_test)
    mock_mse.assert_called_once_with(y_test, mock_model.predict.return_value)
    mock_joblib_dump.assert_called_once_with(mock_model.fitted_pipeline_, 'model.pkl')
