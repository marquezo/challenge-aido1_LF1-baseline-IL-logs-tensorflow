# Definition of Submission container
# We start from tensorflow-gpu image
FROM tensorflow/tensorflow:latest-gpu
#FROM duckietown/rpi-duckiebot-base:master18
#FROM ros:kinetic-ros-base-xenial


# from https://github.com/tensorflow/tensorflow/issues/10776
# this may not be needed if you use pytorch image
RUN cd /usr/local/cuda/lib64 \
    && mv stubs/libcuda.so ./ \
    && ln -s libcuda.so libcuda.so.1 \
    && ldconfig

# DO NOT MODIFY: your submission won't run if you do
RUN apt-get update -y && apt-get install -y --no-install-recommends \
         gcc \
         libc-dev\
         git \
         bzip2 \
         python-tk && \
     rm -rf /var/lib/apt/lists/*

# let's create our workspace, we don't want to clutter the container
RUN mkdir /workspace

# here, we install the requirements, some requirements come by default
# you can add more if you need to in requirements.txt
COPY setup.py /workspace
# Install dependencies
RUN pip install -e /workspace

# let's copy all our solution files to our workspace
# if you have more file use the COPY command to move them to the workspace
COPY src /workspace

# we make the workspace our working directory
WORKDIR /workspace

RUN mkdir /workspace/data
COPY data /workspace/data

RUN ls /workspace
RUN ls
RUN ls /workspace/data


## Download logs
#RUN python download_logs.py
## Extract data into useable format
#RUN python extract_data.py
# Train CNN model for lane following
RUN python cnn_training_tensorflow.py
# Freeze model graph
RUN python freeze_graph.py
