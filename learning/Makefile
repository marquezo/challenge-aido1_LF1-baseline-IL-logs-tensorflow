VIRTUALENV = LF_IL_virtualenv_learning
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

# ---------- DOCKER --------------
# This may run for >10 min and longer if more data is used for learning.
learn-docker:
				docker build -t learn_container .; \
				docker run --runtime=nvidia learn_container;

# Extract data from docker container and copies it to the learning repository
copy_for_submission:
				mkdir ../submission/learned_models; \
				docker cp $(shell docker ps -alq):/workspace/learned_models/frozen_graph.pb ../submission/learned_models;

# ---------- REGULAR --------------
# This assumes that you have GPU drivers installed already.

# If only CPU is available for computation on your machine,
# change the line "pip install tensorflow-gpu" to "tensorflow" (without the GPU option)
install-dependencies:
				virtualenv $(VIRTUALENV); \
				. $(VIRTUALENV)/bin/activate; \
				pip install -e .; \
				pip install tensorflow-gpu;

learn-regular:
				. $(VIRTUALENV)/bin/activate; \
				python src/cnn_training_tensorflow.py; \
				python src/freeze_graph.py;

regular_copy_for_submission:
				mkdir ../submission/learned_models; \
				cp learned_models/frozen_graph.pb ../submission/learned_models;

