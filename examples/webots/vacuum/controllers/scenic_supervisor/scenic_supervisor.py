import sys
import os

try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    scenic_src_path = os.path.normpath(os.path.join(current_script_dir, '..', '..', '..', '..', 'src'))
    if scenic_src_path not in sys.path:
        sys.path.insert(0, scenic_src_path)
except Exception as e:
    print(f"Warning: Could not add Scenic src to path. Error: {e}")

from scenic.gym import ScenicGymEnv
import scenic
from scenic.simulators.webots import WebotsSimulator

import gymnasium as gym
import numpy as np

from controller import Supervisor

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
#-
import matplotlib.pyplot as plt
import time
import gc
from collections import deque

def adjust_clip_range(current_total_coverage_sum: float) -> float:
    input_val = -0.26 * current_total_coverage_sum + 5.2
    # input_val = -0.03 * current_total_coverage_sum + 1
    return input_val

start = time.time()

supervisor = Supervisor()
simulator = WebotsSimulator(supervisor)
print("Webots simulator initialized.")
prefix = scenic.__file__[:-22]

action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))
observation_space = gym.spaces.Dict({
    "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
    "position": gym.spaces.Box(low=np.array([-2.6, -2.6]), high=np.array([2.6, 2.6]), shape=(2,),dtype=np.float64),
    "lidar": gym.spaces.Box(low=0.25, high=5.2, shape=(32,), dtype=np.float64),
    "rotation": gym.spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), shape=(4,), dtype=np.float64)
})
print("Action and observation spaces defined.")
max_steps = 1000

iterations = 6
timesteps_per_itr = max_steps * 1

scenario_template = scenic.scenarioFromFile(prefix + "examples/webots/vacuum/vacuum.scenic",
                                 model="scenic.simulators.webots.model",
                                 mode2D=False)
print("Scenario template loaded.")

training_env_unwrapped = ScenicGymEnv(scenario_template,
                   simulator,
                   render_mode=None,
                   max_steps=max_steps,
                   action_space=action_space,
                   observation_space=observation_space,
                   feedback_fn=adjust_clip_range
                   )
print("Training environment created.")
training_env = Monitor(training_env_unwrapped)

total_timesteps = iterations * timesteps_per_itr

model = PPO("MultiInputPolicy", training_env, verbose=2, learning_rate=0.0002)


# for i in range(iterations):
#     print(f"\n--- Training Progress: Iteration {i+1}/{iterations} ---")

#     model.learn(total_timesteps=timesteps_per_itr)
    
#     model.save("PPO_vacuum_agent_latest")

#     gc.collect()

#------------------------------------------------
model = PPO.load("PPO_vacuum_agent_latest_arsalan.zip", env=training_env)
# print("Loaded model from file.")
model.learn(total_timesteps)
print("Training completed.")
# model.save("PPO_vacuum_agent_latest_ethancodeandlidar")
del model
del training_env
gc.collect()

print(f"\nTraining completed. Total timesteps trained: {total_timesteps}")

print("\n--- Starting Evaluation ---")
scenario_eval = scenic.scenarioFromFile(prefix + "examples/webots/vacuum/vacuum.scenic",
                                 model="scenic.simulators.webots.model",
                                 mode2D=False)

eval_env = ScenicGymEnv(scenario_eval,
                            simulator,
                            render_mode=None,
                            max_steps=max_steps,
                            action_space=action_space,
                            observation_space=observation_space
                            )
eval_env = Monitor(eval_env)

final_model = PPO.load("PPO_vacuum_agent_latest")
final_model.set_env(eval_env)

mean_rwd, std_reward = evaluate_policy(final_model, eval_env, n_eval_episodes=15, render=False, deterministic=False)

print(f"After evaluation mean reward was : {mean_rwd:.2f} with std: {std_reward:.2f}")

episodic_rewards = eval_env.get_episode_rewards()
print(f"Episodic Rewards (Evaluation): {episodic_rewards}")

if len(episodic_rewards) > 1:
    total_pc = 0
    for j in range(1, len(episodic_rewards)):
        if np.abs(episodic_rewards[j - 1]) > 1e-6:
            total_pc += (episodic_rewards[j] - episodic_rewards[j - 1]) / np.abs(episodic_rewards[j - 1])
    print(f"Average normalized percent difference: {total_pc / (len(episodic_rewards) - 1):.4f}")
else:
    print("Not enough episodes for average normalized percent difference calculation.")

fig, ax = plt.subplots()
ax.stem(range(len(episodic_rewards)), episodic_rewards)
plt.title(f"Episodic Rewards During Evaluation (Total Timesteps: {total_timesteps})")
plt.xlabel("Episode Number")
plt.ylabel("Total Reward")
file_name = f"PPO_policy_{total_timesteps}.png"
plt.savefig(file_name, format='png')
plt.show()



end = time.time()
print(f"Total script execution time: {(end - start) / 60:.2f} minutes")
