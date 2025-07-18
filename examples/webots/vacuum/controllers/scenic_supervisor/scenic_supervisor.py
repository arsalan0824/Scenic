import sys
import os

# --- Path Fix for Scenic Imports ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    #  Ensures Python can find Scenic's internal modules.
    scenic_src_path = os.path.normpath(os.path.join(current_script_dir, '..', '..', '..', '..', 'src'))

    if scenic_src_path not in sys.path:
        sys.path.insert(0, scenic_src_path)
except Exception as e:
    print(f"Warning: Could not add Scenic src to path. Error: {e}")
# --- End Path Fix ---

from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.newtonian_gym import NewtonianSimulator
from scenic.simulators.webots import WebotsSimulator
from scipy.stats import qmc

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC,PPO
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

import matplotlib.pyplot as plt
import time

# Dynamic clip_range adjustment function
def adjust_clip_range(current_total_coverage_sum: float) -> float:
    # `current_total_coverage_sum` ranges from 0 (worst coverage) to 100 (best coverage).
    # We map this sum to the lidar's max range using the "100 - sum" concept.

    input_val = 100 - current_total_coverage_sum # equation for clip range Inverse relationship: higher sum -> lower input_val

    min_physical_lidar_range = 0.05 # Shortest lidar range (hardest obstacle detection)
    max_physical_lidar_range = 2.6   # Longest lidar range (easiest obstacle detection)

    # Interpolate the `input_val` (90-100) to the desired physical range (0.05-2.6).
    new_clip_range = np.interp(input_val,
                               [90, 100], # Input range for interpolation
                               [min_physical_lidar_range, max_physical_lidar_range]) # Output range for interpolation

    new_clip_range = np.clip(new_clip_range, min_physical_lidar_range, max_physical_lidar_range) #  Ensure bounds

    return new_clip_range
# --- End Dynamic clip_range adjustment function ---

start = time.time()
 
supervisor = Supervisor() # Collect the Supervisor node from the simulation
simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


prefix = scenic.__file__[:-22]

#  Initial scenario load for TRAINING: pass an initial `clip_range` parameter.
initial_clip_range_for_training = 2.6 # Start with an easier (longer) lidar range
halton_point = qmc.Halton(d=2, scramble=False).random(1) * 5 - 2.5
spawn_x, spawn_y = halton_point[0]
scenario_training = scenic.scenarioFromFile(prefix + "examples/webots/vacuum/vacuum.scenic", #  Renamed scenario
                                 model="scenic.simulators.webots.model",
                                 mode2D=False,
                                 params={"clip_range": initial_clip_range_for_training,
                                         "spawn_x": spawn_x,
                                         "spawn_y": spawn_y
                                 }) #  Pass parameter



action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    #"sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), 
    "position": gym.spaces.Box(low=np.array([-2.6, -2.6]), high=np.array([2.6, 2.6]), shape=(2,),dtype=np.float64),
    "lidar": gym.spaces.Box(low=0.0, high=1, shape=(16,), dtype=np.float64)
})
max_steps = 10000

#  Create the TRAINING environment instance and wrap it.
training_env = ScenicGymEnv(scenario_training, #  Using training scenario
                   simulator, 
                   render_mode=None, 
                   max_steps=max_steps, 
                   action_space=action_space,
                   observation_space=observation_space,
                   feedback_fn=adjust_clip_range) #  Use the dynamic adjustment function
training_env = Monitor(training_env) #  Wrap the training_env with Monitor


episodes= 50
total_timesteps = max_steps * episodes
print(total_timesteps)

model = PPO("MultiInputPolicy", training_env, verbose=2, learning_rate=0.0002,ent_coef=0.05)
# Create an instance of an agent 
#model.set_parameters("PPO_vacuum_agent") # Load the parameters of a previously trained agent
model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
model.save("PPO_vacuum_agent")               # Save the model after training

# Evaluation Scenario Load: fix `clip_range` to a specific value.
fixed_eval_clip_range = 0.05 #  Use the shortest (hardest) lidar range for consistent evaluation
scenario_eval = scenic.scenarioFromFile(prefix + "examples/webots/vacuum/vacuum.scenic", #  Renamed scenario
                                 model="scenic.simulators.webots.model",
                                 mode2D=False,
                                 params={"clip_range": fixed_eval_clip_range}) #  Fixed parameter for eval

# Create the EVALUATION environment instance and wrap it.
eval_env = ScenicGymEnv(scenario_eval, #  Using evaluation scenario
                           simulator,
                           render_mode=None,
                           max_steps=max_steps,
                           action_space=action_space,
                           observation_space=observation_space,
                           feedback_fn=lambda x: fixed_eval_clip_range) #  Fixed feedback for evaluation
eval_env = Monitor(eval_env) #  Wrap the eval_env with Monitor

mean_rwd, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20,render=False, deterministic=False) #  Use eval_env
print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")

episodic_rewards = eval_env.get_episode_rewards() #  Get rewards from eval_env
print(episodic_rewards)
total_pc = 0
for i in range(1, len(episodic_rewards)):
    total_pc += (episodic_rewards[i] - episodic_rewards[i - 1]) / np.abs(episodic_rewards[i - 1])
print("Average normalized percent difference: " + str(total_pc / (len(episodic_rewards) - 1)))

fig,ax = plt.subplots()

ax.stem(range(len(episodic_rewards)), episodic_rewards)

file_name = "PPO_policy" + str(total_timesteps)  + ".png"
plt.savefig(file_name,format='png')
plt.show()

mean_rwd, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=3,render=False) #  Use eval_env

print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")


end = time.time()

print(f" training time was {(end - start) / 60} minutes for {total_timesteps} timesteps")
