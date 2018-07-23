#!/bin/bash

sudo nvidia-docker build --no-cache -t detectronmod .
thispid=$(sudo nvidia-docker run --name=detectronmod -t -d -v ~/data/tests_PrePro/lm_std:/detectron/detectron/datasets/data/coco -p 4000:80 detectronmod)
sudo nvidia-docker exec -it $thispid bash

sudo nvidia-docker container kill $thispid
sudo nvidia-docker container rm $thispid


