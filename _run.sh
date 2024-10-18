#!/bin/bash
source /root/miniforge3/bin/activate pytorch25
source /opt/intel/oneapi/pytorch-gpu-dev-0.5/oneapi-vars.sh 
source /opt/intel/oneapi/pti/0.9/env/vars.sh 

echo "Inference samples ..."
python infer_sample_fp32.py
python infer_sample_amp.py
python infer_sample_compile.py

echo "Training samples ..."
#python train_sample_fp32.py
#python train_sample_amp.py
python train_sample_compile.py

