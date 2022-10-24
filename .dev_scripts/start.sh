#!/usr/bin/env bash

GPU_ID=$1

docker run --rm -it --gpus '"device='$GPU_ID'"' --ipc=host \
    -v /mnt/d/Competitions/yolo_updated/kaggle_det:/workspace \
    -v /mnt/d/Competitions/yolo_updated/kaggle_det/data_wild:/data \
    -v /home/$USER/.cache/torch:/root/.cache/torch \
    -v /etc/timezone:/etc/timezone:ro \
    -v /etc/localtime:/etc/localtime:ro \
    $USER/kaggle