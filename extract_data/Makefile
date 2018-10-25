VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

# This runs in a docker container. Dependencies are taken care of.
# ---------------- DOCKER -----------------------------------

docker_extract_data:
				docker build -t extract_container .; \

# Extract data from docker container and copies it to the learning repository
docker_copy_for_learning:
				docker create -it --name dummy_for_copying extract_container:latest bash; \
				docker cp dummy_for_copying:/workspace/data ../learning; \
				docker rm -fv dummy_for_copying;

# ---------------- REGULAR -----------------------------------
# This assumes that rosbag and other ROS dependencies are installed

install-dependencies:
				virtualenv $(VIRTUALENV); \
				. $(VIRTUALENV)/bin/activate; \
				pip install -e .; \

regular_extract_data:
				. $(VIRTUALENV)/bin/activate; \
				python src/download_logs.py; \
				python src/extract_data.py; \
				rm -rf data/bag_files;

regular_copy_for_learning:
				cp -r data ../learning;