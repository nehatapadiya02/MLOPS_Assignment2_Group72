import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust PYTHONPATH to include the parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.lime_explainability import main as lime_main

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

@patch('src.lime_explainability.best_model_pipeline.predict')
def test_lime_explainability(mock_predict, sample_processed_data, tmp_path):
    # Mock the prediction to return a consistent output
    mock_predict.return_value = [0.55]

    # Redirect the output HTML to a temporary directory
    with patch('builtins.print') as mock_print:
        lime_main()

    # Check if the LIME explanation file was created
    explanation_file = 'lime_explanation.html'
    assert os.path.exists(explanation_file), f"{explanation_file} was not created."

    # Optionally, verify the content of the explanation file
    with open(explanation_file, 'r') as f:
        content = f.read()
        assert "LIME Explanation" in content, "LIME explanation content is incorrect."

    # Clean up the created explanation file
    os.remove(explanation_file)
