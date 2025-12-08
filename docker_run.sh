#!/bin/bash
# Helper script to run the AnyCam Docker container with proper mounts and environment

docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v ~/.cache/torch/hub:/root/.cache/torch/hub \
  -v /home/kalmanm/Documents/thesis:/data/thesis \
  -e DATASETS_ROOT=/data/thesis \
  anycam-env \
  conda run --no-capture-output -n anycam bash