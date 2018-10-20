VIRTUALENV = LF_IL_virtualenv
AG_VERSION = 2018-10-19

# Makefile to structure work on the AI-DO baselines.

get_data:
				docker build -t extract_container .; \
				docker run -d extract_container; \

# Extract data from docker container and copies it to the learning repository
extract_data:
				docker cp $(shell docker ps -alq):/workspace/data ../learning; \