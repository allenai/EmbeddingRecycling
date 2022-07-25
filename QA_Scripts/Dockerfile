
# The base image, which will be the starting point for the Docker image.
# We're using a PyTorch image built from https://github.com/allenai/docker-images
# because PyTorch is really big we want to install it first for caching.
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# This is the directory that files will be copied into.
# It's also the directory that you'll start in if you connect to the image.
WORKDIR ./

# Copy the `requirements.txt` to `/stage/requirements.txt/` and then install them.
# We do this first because it's slow and each of these commands are cached in sequence.
COPY requirements.txt .
RUN pip install opendelta
RUN pip install sklearn
RUN pip install tensorflow

