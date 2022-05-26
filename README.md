<!--
 * @Description: 
 * @Author: CoeusZhang
 * @Date: 2021-04-28 11:21:22
 * @LastEditTime: 2022-05-26 17:49:09
-->
# Introduction
This repo contains code for learning reward functions and state weights from preferences. 
After installation, find the task to run in main.py.
For example, 
```
python -um crowd_pbrl.main --exp_path <path-to-experiments> --gpu_id 0 --split 1 --game BeamRider --task sample_trajectories --method bt
```
Samples some trajectories for reward learning. Make sure there is a config.gin file in your <path-to-experiments>. An example for configuration file is provided as config.gin. A script for how one can run experiments on Enduro is provided as run.py.

# Installation
1. (On windows with 1080Ti) Setup a virtual Python 3.7 environment and install other dependencies. If you are using miniconda,
```
conda create -n crowd_pbrl python==3.7
conda activate crowd_pbrl
conda install cudatoolkit=10.0
conda install cudnn
pip install tensorflow-gpu==1.15
pip install dopamine-rl==3.0.1
pip install ffmpeg-python
pip install scikit-learn
pip install gym[atari]
pip install matplotlib
```

copy atari files
```
python -m atari_py.import_roms <ROM_path>
```

