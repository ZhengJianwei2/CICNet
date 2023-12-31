#!/usr/bin/env bash

gpus=0,1

data_name=LEVIR
net_G=CICNet
split=test
project_name=CICNet-LEVIR
checkpoint_name=best_ckpt.pt

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


