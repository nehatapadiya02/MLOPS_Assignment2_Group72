from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from best_model_pipeline import pipeline as best_model_pipeline

app = Flask(__name__)

#Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('imputer',SimpleImputer(stragegy='median'))
])

@app.route('/predict', method=['POST'])
def predict():
    data = request.get_json() # Expecting data in JSON format
    df = pd.DataFrame(data)
    
    #Apply preprocessing
    X_preprocessed = preprocessing_pipeline.fit_transform(df)
    
    #Make prediction
    predictions = best_model_pipeline.predict(X_preprocessed)
    
    #Return predictions
    return jsonify(predictions.tolist())
    
if __name__ == "__main__":
    app.run(debug=True)