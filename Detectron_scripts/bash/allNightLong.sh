#!/bin/bash

xterm -e "nvidia-smi -l 1" &
xterm -e "htop" &

python2 /home/sthalham/git/Detectron-Tensorflow/tools/train_net.py --cfg /home/sthalham/workspace/Detectron_scripts/Detectron_configs/tless_e2e_faster_R_101_FPN_2.yaml OUTPUT_DIR /home/sthalham/data/T-less_Detectron/output/tless_kinect_all_and_arti10k

sleep 1800

rm /home/sthalham/git/Detectron-Tensorflow/lib/dataset/data/tlessD

ln -s /home/sthalham/data/T-less_Detectron/tlessSplit /home/sthalham/git/Detectron-Tensorflow/lib/dataset/data/tlessD

python2 /home/sthalham/git/Detectron-Tensorflow/tools/train_net.py --cfg /home/sthalham/workspace/Detectron_scripts/Detectron-configs/tless_split OUTOUT_DIR /home/sthalham/data/T-less_Detectron/output/tless_split_correct

sleep 1800

./blender -b /home/sthalham/workspace/Detectron_scripts/blender_scripts/createArtiDepthTEST.blend -P /home/sthalham/workspace/Detectron_scripts/blender_scripts/createArtiDepthScript.py

sleep 600

# python sript for datset creation
# change dataset in Detectron

python2 /home/sthalham/git/Detectron-Tensorflow/tools/train_net.py --cfg /home/sthalham/workspace/Detectron_scripts/Detectron_configs/tless_arti.yaml OUTPUT_DIR /home/sthalham/data/T-less_Detectron/output/tless_kinect_arti_closer
