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

5. **Set up Docker Buildx**:
   docker/setup-buildx-action@v2

6. **Configure AWS Credentials**:
   aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
   aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
   aws-region: ${{ secrets.AWS_REGION }}

7. **Log in to Amazon ECR**:
   id: ecr-login

8. **Build and push Docker image**:
   IMAGE_URI="${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/${{ secrets.ECR_REPOSITORY }}:${{ github.sha }}"
   docker build -t $IMAGE_URI .
   docker push $IMAGE_URI

9. **Check if Lambda function exists**:
   if aws lambda get-function --function-name ${{ secrets.LAMBDA_FUNCTION_NAME }}; then
       echo "exists=true" >> $GITHUB_ENV
   else
       echo "exists=false" >> $GITHUB_ENV
   fi

10. **Create or update Lambda function**:
    if [ "${{ env.exists }}" = "true" ]; then
        aws lambda update-function-code --function-name ${{ secrets.LAMBDA_FUNCTION_NAME }} \
        --image-uri "${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/${{ secrets.ECR_REPOSITORY }}:${{ github.sha }}"
    else
        aws lambda create-function --function-name ${{ secrets.LAMBDA_FUNCTION_NAME }} \
        --package-type Image \
        --code ImageUri="${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com/${{ secrets.ECR_REPOSITORY }}:${{ github.sha }}" \
        --role arn:aws:iam::${{ secrets.AWS_ACCOUNT_ID }}:role/LambdaExecutionRole \
        --timeout 70 \
        --memory-size 512
    fi
