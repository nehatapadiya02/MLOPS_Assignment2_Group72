# Use the official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.10

# Copy app.py and model.pkl to the container
COPY app.py model.pkl ./

# Copy requirements.txt and install dependencies
COPY requirements_docker.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Command to run the Lambda function handler
CMD ["app.lambda_handler"]