import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tpot import TPOTRegressor

def load_data(file_path):
    return pd.read_csv(file_path)

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=10, random_state=42)
    tpot.fit(X_train, y_train)
    y_pred = tpot.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of the best model: {mse}")
    tpot.export('best_model_pipeline.py')
    print("Model training and selection complete. Best model exported to 'best_model_pipeline.py'.")

def main():
    data = load_data('weather_forecast_csv/processed_weatherHistory.csv')

    X = data.drop('Temperature (C)', axis=1)
    y = data['Temperature (C)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_and_evaluate_model(X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
