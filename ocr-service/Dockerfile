# Use the AWS Lambda Python 3.11 base image for fintech-grade performance and security
FROM public.ecr.aws/lambda/python:3.11

# Install Tesseract OCR and its dependencies
RUN yum install -y \
    tesseract \
    tesseract-langpack-spa \
    tesseract-langpack-eng \
    mesa-libGL

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the CMD to the Lambda S3/Textract handler
CMD [ "lambda_handler.handler" ]
