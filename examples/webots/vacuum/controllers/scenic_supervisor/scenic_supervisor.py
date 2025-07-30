from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C,PPO

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO

import matplotlib.pyplot as plt
import time
import gc

start = time.time()
supervisor = Supervisor() # Collect the Supervisor node from the simulation
prefix = scenic.__file__[:-22]

simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node

action_space = gym.spaces.MultiDiscrete([3,4])  # Defines the possible actions of the agent
observation_space = gym.spaces.Dict({
    "velocity":  gym.spaces.Discrete(16),
    "sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
    "position": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "rotation": gym.spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), shape=(4,), dtype=np.float64)
})
                             
max_steps = 10000
episodes  = 100
total_timesteps = max_steps * episodes

scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
                                model="scenic.simulators.webots.model",
                                mode2D=False)
             

env = Monitor(ScenicGymEnv(scenario, 
                simulator, 
                render_mode=None, 
                max_steps=max_steps, 
                action_space=action_space,
                observation_space=observation_space)) # max_step is max step for an episode - Create an enviroment instance


model = RecurrentPPO("MultiInputLstmPolicy", env,seed=20,ent_coef=0.05)
#model = PPO.load("PPO_vacuum_agent_H50_R100_R50Full", env=env) # Create an instance of an agent 
#model = PPO.load("PPO_vacuum_agent_entropy_test", env=env)
model.learn(total_timesteps=total_timesteps)
model.save("recurrent_PPO_test")

#env.close() #avoid closing the env as it will destroy the simulator instance aswell - won't accomplish what is needed here anyways
# Cleanup to avoid assertion error when instantiating new scenario
#del model
#del env
#gc.collect()

mean_rwd, std = evaluate_policy(model, env=env, n_eval_episodes=10, deterministic=True)

print(f"Mean reward over 10 episodes :{mean_rwd} \n std was: {std}")

episodic_rewards = env.get_episode_rewards()
fig,ax = plt.subplots()

ax.stem(range(len(episodic_rewards)), episodic_rewards)

file_name = "../episode_rewards/PPO_policy_" + str(total_timesteps)  + ".png"
plt.savefig(file_name,format='png')
plt.show()
