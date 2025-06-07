from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import os

from gymnasium.utils.env_checker import check_env

from controller import Supervisor,robot


env = gym.make("CartPole-v1")