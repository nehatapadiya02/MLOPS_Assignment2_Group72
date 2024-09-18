import pandas as pd
import numpy as np
import sweetviz as sv
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer,TargetEncoder,MinMaxScaler

# Load dataset
data = pd.read_csv('weather_forecast_csv/weatherHistory.csv')

# Display basic info
print("Initial Data Info:")
print(data.info())
print("Initial Data Description:")
print(data.describe())

# AutoEDA using Sweetviz
def perform_autoeda(df):
    # Create the report
    report = sv.analyze(df)
    # Save the report as a HTML file
    report.show_html('eda_report.html')
    
perform_autoeda(data)

# Data Cleaning
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
    
    # Detect and remove outliers
    # def remove_outliers(series):
    #     return series[np.abs(series - series.mean()) < (3 * series.std())]
    
    # df['Temperature (C)'] = remove_outliers(df['Temperature (C)'])
    # df['Humidity'] = remove_outliers(df['Humidity'])
    # df['Wind Speed (km/h)'] = remove_outliers(df['Wind Speed (km/h)'])
    
    return df

data = clean_data(data)


def format_and_convert_date(df):
    # Convert 'Formatted Date' to datetime
    df['Formatted Date'] = pd.to_datetime(df['Formatted Date'],utc=True)
    df['Hour'] = df['Formatted Date'].dt.hour
    df['DayOfWeek'] = df['Formatted Date'].dt.dayofweek
    df['Month'] = df['Formatted Date'].dt.month
    return df

data = format_and_convert_date(data)

# Scaling/Normalization and One-Hot Encoding
def preprocess_features(df):
    df = pd.get_dummies(df, columns=['Precip Type'])
    df['TempDiff'] = df['Temperature (C)'] - df['Apparent Temperature (C)']
    mean_target = df.groupby('Summary')['Temperature (C)'].mean()
    df['Summary_encoded'] = df['Summary'].map(mean_target)
    minmax_scaler = MinMaxScaler()
    df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']] = minmax_scaler.fit_transform(df[['Temperature (C)', 'Apparent Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']])
    return df

data = preprocess_features(data)

print("Data Info after pre-processing:")
print(data.info())
print("Data Description after pre-processing:")
print(data.describe())

# Scaling/Normalization and One-Hot Encoding
# def preprocess_features(df):
#     numeric_features = ['Temperature (C)', 'Humidity', 'Wind Speed (km/h)']
#     categorical_features = ['Precip Type']
    
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='median')),
#         ('scaler', StandardScaler())
#     ])
    
#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])
    
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ]
#     )
    
#     X = preprocessor.fit_transform(df)
#     # Get feature names
#     feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['Precip Type']))
    
#     return X, feature_names

# X, feature_names = preprocess_features(data)


feature_names = ['Temperature (C)','Humidity','Wind Speed (km/h)','Visibility (km)','Pressure (millibars)','Hour','DayOfWeek','Month','Precip Type_rain','Precip Type_snow','TempDiff','Summary_encoded']
# Save processed data to a new CSV
processed_df = pd.DataFrame(data, columns=feature_names)
processed_df.to_csv('weather_forecast_csv/processed_weatherHistory.csv', index=False)

print("Data preprocessing complete. Processed data saved to 'weather_forecast_csv/processed_weatherHistory.csv'.")
