VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

learn:
				docker build -t learn_container .; \
				docker run -d learn_container; \

# Extract data from docker container and copies it to the learning repository
copy_for_submission:
				docker cp $(shell docker ps -alq):/workspace/learned_models ../submission; \