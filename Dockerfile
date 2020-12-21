# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
# Install any needed packages specified in requirements.txt
#RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -r requirements.txt



# Run test.py when the container launches
CMD ["python", "docker_test_tfGPU.py"]
