# Set base image
FROM python:3.8-slim

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8



# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1


WORKDIR /app
COPY . /app

# Install Poetry (system dependecy)
RUN pip3 install poetry

# Install poetry dependencies
RUN poetry install

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser


CMD ["python", "src/train_topic_modeller_script.py"]


