import pytest
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust PYTHONPATH to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.shap_explainability import main as shap_main

@pytest.fixture
def sample_processed_data(tmp_path):
    # Create a small sample processed dataset
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
    csv_path = tmp_path / "processed_weatherHistory.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

@patch('src.shap_explainability.best_model_pipeline.predict')
def test_shap_explainability(mock_predict, sample_processed_data, tmp_path):
    # Mock the prediction to return a consistent output
    mock_predict.return_value = [0.55, 0.65]

    # Redirect the output plot to a temporary directory
    with patch('builtins.print') as mock_print:
        shap_main()

    # Check if the SHAP summary plot file was created
    plot_file = 'shap_summary_plot.png'
    assert os.path.exists(plot_file), f"{plot_file} was not created."

    # Optionally, verify the content of the plot file
    assert os.path.getsize(plot_file) > 0, "SHAP summary plot file is empty."

    # Clean up the created plot file
    os.remove(plot_file)
