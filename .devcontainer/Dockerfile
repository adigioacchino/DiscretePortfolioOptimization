# Use the official Python 3.13 image as the base image
FROM python:3.13-slim

# install git
RUN apt-get update && apt-get install -y git

# install curl
RUN apt-get update && apt-get install -y curl

# Install the 'uv' package
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
