# MLOPS_Assignment2_Group72

## Project Overview

This project involves data preprocessing, model training, and explainable AI (XAI) for weather forecast prediction. The following tasks are included:

1. **Data Collection and Preprocessing**:
    - Data Cleaning
    - Feature Engineering
    - Scaling and normalization
    - AutoEDA using Sweetviz

2. **Model Selection, Training, and Hyperparameter Tuning**:
    - Training multiple models
    - Hyperparameter tuning using TPOT
    - Model evaluation and selection

3. **Explainable AI (XAI)**:
    - Model Interpretability using SHAP
    - Local explanations using LIME

## Files

- `weather_forecast_csv`:
   - `weatherHistory.csv`: Original dataset
   - `processed_weatherHistory.csv`: Processed dataset

- `src/`:
   - `data_preprocessing.py`: Script for data preprocessing
   - `model_selection.py`: Script for model training and evaluation
   - `shap_explainability.py`: Script for SHAP model explainations
   - `lime_explainability.py`: Script for LIME model explainations

- `best_model_pipeline.py`: Exported best model pipeline from TPOT
- `requirements.txt`: List of dependencies
- `README.md`: Project documentation

## How to run

1. **Preprocess Data**:
    Run the data preprocessing script:
    ```sh
    python src/data_preprocessing.py

2. **Train and Evaluate Model**:
    python src/model_selection.py

3. **Generate SHAP explainations**:
    python src/shap_explainability.py

4. **Generate LIME explanations**:
    python src/lime_explainability.py

## Dependencies
pip install -r requirements.txt
