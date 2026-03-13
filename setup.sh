#!/bin/bash
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install nvcc4jupyter numpy jupyter_client ipykernel ipywidgets
pip install triton
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install matplotlib
pip install pandas
pip install nvidia-cutlass-dsl
pip install pufferlib --no-build-isolation
pip install stable-baselines3 
pip install wandb
git clone https://github.com/PufferAI/PufferLib.git


git config --global user.email sathya.pranav.deepak@gmail.com
git config --global user.name PranavDeepakSathya
