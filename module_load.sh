#!/bin/bash
module purge
module load anaconda3_gpu
module load cuda
module load cudnn
conda activate jax
