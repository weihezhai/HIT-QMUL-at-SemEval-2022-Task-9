#!/bin/bash
#$ -l h_rt=240:0:0
#$ -l h_vmem=7.5G
#$ -pe smp 32
#$ -l gpu=4
#$ -l gpu_type=volta
#$ -cwd
#$ -j y
#$ -N job
#$ -m bea
#$ -l h=sbg3
source ~/pytorchenv/bin/activate
python ./T5.py
