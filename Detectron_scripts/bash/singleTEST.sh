#!/bin/bash

xterm -e "nvidia-smi -l 1" &
xterm -e "htop" &

python2 /home/sthalham/git/Detectron-Tensorflow/tools/train_net.py --cfg /home/sthalham/workspace/Detectron_scripts/Detectron_configs/tlessArti03052018.yaml OUTPUT_DIR /home/sthalham/data/T-less_Detectron/output/tlessArti03052018

