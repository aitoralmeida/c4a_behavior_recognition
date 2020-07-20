#!/bin/bash

sudo docker stop ubermejo-behaviour-recognition
sudo docker rm ubermejo-behaviour-recognition
sudo docker run -it --ipc=host --gpus "device=0" --name ubermejo-behaviour-recognition -v behaviour-recognition:/results ubermejo/behaviour_recognition bin/bash
