# lambda/predict/app.py

import json
import joblib
import numpy as np
import os

# joblib.parallel_backend('loky')  # Use 'loky' to prevent using multiprocessing

        # Load the model from the same directory as this script
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

def lambda_handler(event, context):
    try:
        # Parse the incoming JSON data
        body = json.loads(event['body'])
        features = body.get('features')
        
        if not features:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Please provide features in the request body.'})
            }
        
        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        
        # Return the prediction
        return {
            'statusCode': 200,
            'body': json.dumps({'prediction': prediction.tolist()})
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
