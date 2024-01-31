#!/bin/bash

DOCKER_NAME=multicorrupt_create

# path to directory where nusenes data is stored
multicorrupt_data_dir="/work/multicorrupt"
nuscenes_data_dir="/work/nuscenes"

# path to this repository root
repo_dir=$PWD

docker run \
--name data_create_container \
--rm \
--gpus 'all,"capabilities=compute,utility,graphics"' \
--env DISPLAY=${DISPLAY} \
--shm-size=16gb \
--net=host \
--user root \
--volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
--volume $HOME/.Xauthority:/.Xauthority:rw \
--mount source=$repo_dir,target=/workspace,type=bind,consistency=cached \
--mount source=$multicorrupt_data_dir,target=/workspace/multicorrupt,type=bind,consistency=cached \
--mount source=$nuscenes_data_dir,target=/workspace/nuscenes,type=bind,consistency=cached \
-it \
-d \
$DOCKER_NAME

