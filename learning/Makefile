VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

# ---------- DOCKER --------------
learn-docker:
				docker build -t learn_container .; \
				docker run --runtime=nvidia learn_container;

# Extract data from docker container and copies it to the learning repository
copy_for_submission:
				docker cp $(shell docker ps -alq):/workspace/learned_models ../submission; \

# ---------- REGULAR --------------
install-dependencies:
				virtualenv $(VIRTUALENV); \
				. $(VIRTUALENV)/bin/activate; \
				pip install -e .;

learn-regular:
				. $(VIRTUALENV)/bin/activate; \
				python src/cnn_training_tensorflow.py; \
				python src/freeze_graph.py;

