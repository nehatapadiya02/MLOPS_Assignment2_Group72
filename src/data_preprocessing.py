import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder  # Ensure this is installed

def perform_autoeda(df):
    # Create the report
    report = sv.analyze(df)
    # Save the report as an HTML file
    report.show_html('eda_report.html')

def clean_data(df):
    # Handling missing values
    df = df.fillna({
        'Temperature (C)': df['Temperature (C)'].median(),
        'Humidity': df['Humidity'].median(),
        'Precip Type': df['Precip Type'].mode()[0],  # Use mode for categorical data
        'Wind Speed (km/h)': df['Wind Speed (km/h)'].median()
    })

    # Removing duplicates
    df = df.drop_duplicates()
    return df

def format_and_convert_date(df):
    # Convert 'Formatted Date' to datetime
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
    df['Hour'] = df['Formatted Date'].dt.hour
    df['DayOfWeek'] = df['Formatted Date'].dt.dayofweek
    df['Month'] = df['Formatted Date'].dt.month
    return df

def preprocess_features(df):
    df = pd.get_dummies(df, columns=['Precip Type'])
    df['TempDiff'] = df['Temperature (C)'] - df['Apparent Temperature (C)']
    mean_target = df.groupby('Summary')['Temperature (C)'].mean()
    df['Summary_encoded'] = df['Summary'].map(mean_target)
    minmax_scaler = MinMaxScaler()
    df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']] = minmax_scaler.fit_transform(
        df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']]
    )
    return df

def main():
    # Load dataset
    data = pd.read_csv('weather_forecast_csv/weatherHistory.csv')

    # Display basic info
    print("Initial Data Info:")
    print(data.info())
    print("Initial Data Description:")
    print(data.describe())

    # AutoEDA using Sweetviz
    perform_autoeda(data)

    # Data Cleaning
    data = clean_data(data)

    # Format and convert date
    data = format_and_convert_date(data)

    # Preprocess features
    data = preprocess_features(data)

    print("Data Info after pre-processing:")
    print(data.info())
    print("Data Description after pre-processing:")
    print(data.describe())

    feature_names = [
        'Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)',
        'Pressure (millibars)', 'Hour', 'DayOfWeek', 'Month',
        'Precip Type_rain', 'Precip Type_snow', 'TempDiff', 'Summary_encoded'
    ]
    # Save processed data to a new CSV
    processed_df = pd.DataFrame(data, columns=feature_names)
    processed_df.to_csv('weather_forecast_csv/processed_weatherHistory.csv', index=False)

    print("Data preprocessing complete. Processed data saved to 'weather_forecast_csv/processed_weatherHistory.csv'.")

if __name__ == "__main__":
    main()
