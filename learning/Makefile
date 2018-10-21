VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

learn-docker:
				docker build -t learn_container .; \
				docker run -d learn_container; \

learn-regular:
				python src/cnn_training_tensorflow.py; \
				python src/freeze_graph.py;

# Extract data from docker container and copies it to the learning repository
copy_for_submission:
				docker cp $(shell docker ps -alq):/workspace/learned_models ../submission; \