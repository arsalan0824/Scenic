# #pasted ethans code, and added all my contributions
# import sys
# import os

# try:
#     current_script_dir = os.path.dirname(os.path.abspath(__file__))
#     scenic_src_path = os.path.normpath(os.path.join(current_script_dir, '..', '..', '..', '..', 'src'))
#     if scenic_src_path not in sys.path:
#         sys.path.insert(0, scenic_src_path)
# except Exception as e:
#     print(f"Warning: Could not add Scenic src to path. Error: {e}")

# from scenic.gym import ScenicGymEnv
# import scenic
# from scenic.simulators.newtonian_gym import NewtonianSimulator
# from scenic.simulators.webots import WebotsSimulator

# import gymnasium as gym
# import numpy as np

# from controller import Supervisor

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import SAC,PPO
# from stable_baselines3.common.monitor import Monitor


# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.evaluation import evaluate_policy

# import matplotlib.pyplot as plt
# import time
# import gc
# from collections import deque

# # class GAECallback(BaseCallback):
# #     def _on_step(self) -> bool:
# #         # SB3 makes a `dones` array available here
# #         dones = self.locals.get("dones")
# #         # whenever any env signals done=True, an episode just ended
# #         if dones is not None and np.any(dones):
# #             # 1) grab the last rollout's per-step GAE
# #             advantages = self.model.rollout_buffer.advantages  # shape (n_steps * n_envs,)
# #             # 2) turn it into one scalar (sum of absolutes is common)
# #             priority = np.average(np.abs(advantages))

# #             # 3) find your ScenicGymEnv inside the VecEnv/Monitor wrappers
# #             envs = getattr(self.training_env, "envs", [self.training_env])
# #             for e in envs:
# #                 real = getattr(e, "env", e)    # unwrap Monitor if present
# #                 if isinstance(real, ScenicGymEnv):
# #                     idx = real.working_index
# #                     # overwrite the PLR buffer entry for this scene
# #                     if idx < len(real.buffer_learning_potential):
# #                         real.buffer_learning_potential[idx] = priority
# #                     else:
# #                         real.buffer_learning_potential = np.append(
# #                             real.buffer_learning_potential, priority
# #                         )
# #         return True 
# # gae_cb = GAECallback()

# def adjust_clip_range(current_total_coverage_sum: float) -> float:
#     input_val = -0.26 * current_total_coverage_sum + 5.2
#     return input_val

# start = time.time()

# supervisor = Supervisor() # Collect the Supervisor node from the simulation
# simulator = WebotsSimulator(supervisor) # Create an instance of the WebotsSImulator with the corresponding node


# prefix = scenic.__file__[:-22]
# scenario = scenic.scenarioFromFile(prefix +  "examples/webots/vacuum/vacuum.scenic",
#                                    model="scenic.simulators.webots.model",
#                                    mode2D=False) # generate the scenario from the corresponding Scenic file



# action_space = gym.spaces.Box(low=-1.0, high=1.0 ,shape=(2,))  # Defines the possible actions of the agent
# array_size = 1 #find in simulator.py by ctrl f'ing array_size
# observation_space = gym.spaces.Dict({
#     "velocity": gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), shape=(2,),dtype=np.float64),
#     #"sensor": gym.spaces.Box(low=np.array([0,0,0,0,0,0,0]), high=np.array([1,1,1,1,1,1,1]),shape=(7,),dtype=np.float64), # defines the range of observations of the agent
#     "position": gym.spaces.Box(low=np.array([-2.6, -2.6]), high=np.array([2.6, 2.6]), shape=(2,),dtype=np.float64),
#     "lidar": gym.spaces.Box(low=0.01, high=1, shape=(36,), dtype=np.float64),
#     "rotation": gym.spaces.Box(low=np.array([-1,-1,-1,-1]), high=np.array([1,1,1,1]), shape=(4,), dtype=np.float64)
#     #"sectional_coverage": gym.spaces.Box(low=np.zeros(16), high=np.ones(16), shape=(16,),dtype=np.float64),
#     # "current_section": gym.spaces.Box(low=np.array([0]), high=np.array([15]), shape=(1,),dtype=int)
# })
# max_steps = 10000
# env = ScenicGymEnv(scenario, 
#                    simulator, 
#                    render_mode=None, 
#                    max_steps=max_steps,  
#                    action_space=action_space,
#                    observation_space=observation_space) # max_step is max step for an episode - Create an enviroment instance
# env = Monitor(env)

# episodes= 21
# total_timesteps = max_steps * episodes
# print(total_timesteps)

# training_env_unwrapped = ScenicGymEnv(scenario_template,
#                    simulator,
#                    render_mode=None,
#                    max_steps=max_steps,
#                    action_space=action_space,
#                    observation_space=observation_space,
#                    feedback_fn=adjust_clip_range
#                    )
# training_env = Monitor(training_env_unwrapped)

# model = PPO("MultiInputPolicy", env, verbose=2, learning_rate=0.0002,ent_coef=0.05)
# # Create an instance of an agent 
# model.set_parameters("baseline.zip") # Load the parameters of a previously trained agent
# # model.learn(total_timesteps=total_timesteps)          # train the agent over a set number of steps
# #model.save("7_21 - 50 + 50 VerifAI")               # Save the model after training

# #mean_rwd, std_reward = evaluate_policy(model, env, n_eval_episodes=10,render=False, deterministic=False)
# #print(f"After evaluation mean reward was : {mean_rwd} with std: {std_reward}")
# for i in range(iterations):
#     print(f"\n--- Training Progress: Iteration {i+1}/{iterations} ---")

#     model.learn(total_timesteps=timesteps_per_itr)
    
#     model.save("PPO_vacuum_agent_latest")

#     gc.collect()

# total_timesteps = iterations * timesteps_per_itr
# print(f"\nTraining completed. Total timesteps trained: {total_timesteps}")

# print("\n--- Starting Evaluation ---")
# scenario_eval = scenic.scenarioFromFile(prefix + "examples/webots/vacuum/vacuum.scenic",
#                                  model="scenic.simulators.webots.model",
#                                  mode2D=False)

# eval_env = ScenicGymEnv(scenario_eval,
#                             simulator,
#                             render_mode=None,
#                             max_steps=max_steps,
#                             action_space=action_space,
#                             observation_space=observation_space
#                             )
# eval_env = Monitor(eval_env)

# final_model = PPO.load("Lidar_PPO_base_50_mod.zip")
# final_model.set_env(eval_env)

# mean_rwd, std_reward = evaluate_policy(final_model, eval_env, n_eval_episodes=50, render=False, deterministic=False)

# print(f"After evaluation mean reward was : {mean_rwd:.2f} with std: {std_reward:.2f}")

# episodic_rewards = eval_env.get_episode_rewards()
# print(f"Episodic Rewards (Evaluation): {episodic_rewards}")

# env.env.logScores()

# episodic_rewards = env.get_episode_rewards()
# print(episodic_rewards)
# total_pc = 0
# if(len(episodic_rewards) >= 2):
#     for i in range(1, len(episodic_rewards)):
#         total_pc += (episodic_rewards[i] - episodic_rewards[i - 1]) / np.abs(episodic_rewards[i - 1])
#     print("Average normalized percent difference: " + str(total_pc / (len(episodic_rewards) - 1)))

# fig,ax = plt.subplots()

# ax.stem(range(len(episodic_rewards)), episodic_rewards)

# file_name = "PPO_policy" + str(total_timesteps)  + ".png"
# plt.savefig(file_name,format='png')
# plt.show()


# end = time.time()

# print(f" training time was {(end - start) / 60} minutes for {total_timesteps} timesteps")