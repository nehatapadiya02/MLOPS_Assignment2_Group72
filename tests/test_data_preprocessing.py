import pytest
import pandas as pd
from src.data_preprocessing import clean_data, format_and_convert_date, preprocess_features

@pytest.fixture
def sample_data():
    data = {
        'Formatted Date': ['2020-01-01 00:00:00', '2020-01-01 01:00:00'],
        'Temperature (C)': [20.0, 21.0],
        'Apparent Temperature (C)': [19.0, 20.0],
        'Humidity': [0.80, 0.85],
        'Wind Speed (km/h)': [10, 12],
        'Visibility (km)': [10, 10],
        'Pressure (millibars)': [1015, 1016],
        'Precip Type': ['rain', 'snow'],
        'Summary': ['Clear', 'Cloudy']
    }
    df = pd.DataFrame(data)
    return df

def test_clean_data(sample_data):
    cleaned_df = clean_data(sample_data.copy())
    # Check for no missing values
    assert cleaned_df.isnull().sum().sum() == 0
    # Check duplicates are removed (none in sample data)
    assert len(cleaned_df) == len(sample_data)

def test_format_and_convert_date(sample_data):
    formatted_df = format_and_convert_date(sample_data.copy())
    # Check new columns
    assert 'Hour' in formatted_df.columns
    assert 'DayOfWeek' in formatted_df.columns
    assert 'Month' in formatted_df.columns
    # Check datetime conversion
    assert pd.api.types.is_datetime64_any_dtype(formatted_df['Formatted Date'])

def test_preprocess_features(sample_data):
    processed_df = preprocess_features(sample_data.copy())
    # Check new columns
    expected_columns = [
        'Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)',
        'Visibility (km)', 'Pressure (millibars)', 'Hour', 'DayOfWeek', 'Month',
        'Precip Type_rain', 'Precip Type_snow', 'TempDiff', 'Summary_encoded'
    ]
    for col in expected_columns:
        assert col in processed_df.columns
    # Check scaling (values between 0 and 1)
    scaled_columns = ['Temperature (C)', 'Apparent Temperature (C)', 'Humidity',
                      'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']
    for col in scaled_columns:
        assert processed_df[col].min() >= 0
        assert processed_df[col].max() <= 1
