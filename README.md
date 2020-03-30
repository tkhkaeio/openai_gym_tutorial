# OpenAI Gym Tutorial



## Requirements

Python>=3.6
TensorFlow==1.14.0
OpenAI Gym==0.15.3
Gym Retro==0.7.0
Stable Baselines==2.8.0


## Installation

for mac
```
# openai gym
brew install cmake boost boost-python sdl2 swig wget
pip install gym
# specify env name in []
pip install gym[atari]
pip install gym[box2d]

# stable baselines
brew install cmake openmpi
pip install stable-baselines[mpi]
pip install tesorflow==1.14.0
pip instal pyqt5
pip install imageio
```

for ubuntu
```
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
pip install gym
# specify env name in []
pip install gym[atari]
pip install gym[box2d]

# stable baselines
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
pip install stable-baselines[mpi]
pip install tesorflow==1.14.0
pip instal pyqt5
pip install imageio
```

Note: need to buy license for mujoco, robotics environments

## Stable Baselines
collection of RL models' implementations, including A2C, ACER, ACKTR, DDPG, DQN, GAIL, HER, PPO1, PPO2, SAC, TD3, TRPO

other RL libraries
- Coach
- RLLib
- Dopamine

## Stable Baselines Zoo
collection of pretrained models in stable baselines, with best hyperparameter settings
```
pip install stable-baselines pyyaml pybullet optuna pytablewriter
```

Refer to the repo for more sophisticated code and hyperparameters \
https://github.com/araffin/rl-baselines-zoo


