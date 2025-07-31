# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container
COPY ./requirements.txt /code/requirements.txt

# Install the packages specified in requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy your application code into the container
COPY ./app.py /code/app.py

# Expose the port the app runs on (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Command to run your application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]