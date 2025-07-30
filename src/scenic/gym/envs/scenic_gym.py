from scenic.core.simulators import Simulator, Simulation, TerminationType
from scenic.core.errors import setDebuggingOptions
from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable
import random
import pandas as pd
from collections import deque
import numpy as np

setDebuggingOptions(verbosity=2)

#TODO make ResetException

file_path = "../../../../../output.csv"
point_file_path = "../../../../../points.csv"

def write_csv(name, coverage, collisions, rewards):
    rows = [[f"coverage_{name}"] + list(coverage),
            [f"collisions_{name}"] + list(collisions),
            [f"rewards_{name}"] + list(rewards)
            ]
    df = pd.DataFrame(rows)
    df.to_csv(file_path, index=False, mode='a', header=False)
    
def write_point_records(name, timewise_points):
    rows = [[f"{name}"] + list(timewise_points)]
    df = pd.DataFrame(rows)
    df.to_csv(point_file_path, index=False, mode='a',header=False)

# Custom exception class
class ResetException(Exception):
    def __init__(self):
        super().__init__("Resetting")

class ScenicGymEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self,
                 scenario : Scenario,
                 simulator : Simulator,
                 render_mode=None,
                 max_steps = 1000,
                 observation_space : spaces.Dict = spaces.Dict(),
                 action_space : spaces.Dict = spaces.Dict(),
                 record_scenic_sim_results : bool = True,
                 feedback_fn : Callable = lambda x: x):

        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.max_steps = max_steps - 1
        self.simulator = simulator
        self.scenario = scenario
        self.simulation_results = []

        self.feedback_result = None
        self.loop = None
        self.record_scenic_sim_results = record_scenic_sim_results
        self.feedback_fn = feedback_fn

        self.training_method = "Random"
        self.use_plr = False
        self.use_verifai = False
        self.buffer_p = 0.5 # probability of resampling


        self.save_to_csv = True # whether to save data to csv
        self.record_points = True
        self.run_name = "7_29 - F50, ent(.025)" # name of the run, used for saving data to csv
        self.total_steps = 10000*50 # total number of timesteps to run, used for saving data to csv

        self.resampling_weights =  np.array([]) # resamling weights of scenes in the buffer
        self.buffer_last_reward = np.array([]) # last reward of the scenes in the buffer(for lp)
        
        #extra variables for the run loop
        self.working_index = -1
        self.flag = 0
        self.counting_reward = 0
        self.steps_taken = 0
        self.total_steps_taken = 0
    
        if self.use_plr and self.training_method not in ("EL", "LP"):
            raise ValueError(
                f"use_plr=True but training_method={self.training_method!r}. "
                "Must be one of 'LP' or 'EL' if use_plr is enabled."
            )
        
        #information recording
        self.episode_coverages = []  
        self.episode_collisions = []  
        self.episode_rewards = []
        self.timewise_points = None

        self.last_10_episode_coverages = deque(maxlen=10)
        self.total_episodes_completed = 0
        self.current_total_coverage_sum = 0

    def _make_run_loop(self): 
        while True:
            try:
                scene = None
                self.is_resampling = random.uniform(0, 1) < self.buffer_p
                #sample or resample scenes
                if not self.use_plr:
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                elif self.training_method == "LP" and len(self.resampling_weights) != len(self.buffer_last_reward):
                    # resample scene
                    self.flag = 0
                    self.working_index = len(self.buffer_last_reward) - 1
                    with open(f"../../../../../../buffer/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print("Double sampling")
                elif self.use_plr and self.is_resampling and len(self.resampling_weights) > 0:
                    # resample from buffer
                    self.flag = 1
                    prob_distribution = self.resampling_weights / np.sum(self.resampling_weights)
                    self.working_index = np.random.choice(len(self.resampling_weights), p=prob_distribution)
                    with open(f"../../../../../../buffer/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print(f"Resampling from buffer with index {self.working_index}")
                else:
                    # sample new scene
                    self.flag = 2
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                    with open(f"../../../../../../buffer/scene_{len(self.resampling_weights)}.bin", "wb") as f:
                        f.write(self.scenario.sceneToBytes(scene=scene))
                    self.working_index = len(self.resampling_weights)
                    print(f"Sampling new scene with index {self.working_index}")
                
                self.counting_reward = 0

#---------------------------------------------

                scene, _ = self.scenario.generate(feedback=self.feedback_result)
                #make a variable so self.current_total_coverage_sum is accessible in simulator.py
                
                with self.simulator.simulateStepped(scene, maxSteps=self.max_steps) as simulation:
                    self.steps_taken = 0

                    simulation.current_total_coverage_sum = self.current_total_coverage_sum
                    steps_taken = 0
                    done_episode = lambda: not (simulation.result is None) or (simulation.get_truncation())
                    truncated_episode = lambda: (steps_taken >= self.max_steps)

                    observation = simulation.get_obs()
                    initial_info = {}
                    actions = yield observation, initial_info
                    simulation.actions = actions # TODO add action dict to simulation interfaces

                    while not done_episode():
                        simulation.actions = actions
                        reward, step_info = simulation.get_reward()

                        simulation.advance()
                        steps_taken += 1

                        observation = simulation.get_obs()
                        current_info = simulation.get_info()
                        current_info.update(step_info)

                        if done_episode() or truncated_episode():
                            _, coverage_ratio = simulation.get_coverage_metric()
                            self.last_10_episode_coverages.append(coverage_ratio)

                            self.total_episodes_completed += 1
                            self.current_total_coverage_sum = np.sum(self.last_10_episode_coverages)

                            # Condition for printing episode coverage summary and updating feedback
                            if self.current_total_coverage_sum > 5:
                                print ("lidar has passed 50%, clipping range will be adjusted")
                                # Print episode coverage summary
                                print(f"Episode {self.total_episodes_completed}: "
                                      f"Sum of last {len(self.last_10_episode_coverages)} "
                                      f"episode coverages: {self.current_total_coverage_sum:.5f}")

                                # Call feedback_fn to get the new clip range value
                                if self.feedback_fn is not None:
                                    self.feedback_result = self.feedback_fn(self.current_total_coverage_sum)
                                    # Print the new lidar max range directly from here
                                    print(f"new LIDAR max range is: {self.feedback_result:.3f} meters")
                            # For episodes NOT divisible by 10, ensure feedback_result is still set
                            # for subsequent scenario generations if it hasn't been yet.
                            elif self.feedback_fn is not None and self.feedback_result is None:
                                self.feedback_result = self.feedback_fn(self.current_total_coverage_sum)
                            else:
                                print("Feedback function is None, no new clip range set.")

                            if self.record_scenic_sim_results:
                                self.simulation_results.append(simulation.result)

                            actions = yield observation, reward, done_episode(), truncated_episode(), current_info
                            break

                        actions = yield observation, reward, done_episode(), truncated_episode(), current_info

            except ResetException:
                if self.total_steps_taken >= self.total_steps and self.save_to_csv:
                    write_csv(self.run_name, self.episode_coverages, self.episode_collisions, self.episode_rewards)
                print("reset exception caught")
                print(f"Episode coverages: {self.episode_coverages}")
                print(f"Mean and std of coverages: {np.mean(self.episode_coverages)} and {np.std(self.episode_coverages)}")
                print(f"Episode collisions: {self.episode_collisions}")
                print(f"Mean and std of collisions: {np.mean(self.episode_collisions)} and {np.std(self.episode_collisions)}")
                print(f"Excel splittable: {np.mean(self.episode_coverages)},{np.std(self.episode_coverages)},{np.mean(self.episode_collisions)},{np.std(self.episode_collisions)}")

                continue

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.loop is None:
            self.loop = self._make_run_loop()
            observation, info = next(self.loop)
        else:
            observation, info = self.loop.throw(ResetException())
        return observation, info

    def step(self, action):
        assert self.loop is not None, "self.loop is None, have you called reset()?"
        observation, reward, terminated, truncated, info = self.loop.send(action)
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        if self.loop is not None:
            try:
                self.loop.close()
            except StopIteration:
                pass
            self.loop = None

    def logScores(self):
        if self.training_method == "Random":
            return
        total_reward = self.counting_reward
        if(total_reward == 0):
            print("TOTAL REWRAD is 0! suspiciosu!")
        #log rewards and learning potential
        if self.flag == 0:
            #double sampling, so we know we are using LP
            if(self.working_index >= len(self.buffer_last_reward)):
                print(f"Warning: working index {self.working_index} is out of bounds for buffer_last_reward with length {len(self.buffer_last_reward)}")
            else:
                lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
                if self.use_verifai:
                    self.feedback_result = -lp
                self.resampling_weights = np.append(self.resampling_weights, lp)
                self.buffer_last_reward[self.working_index] = total_reward
                print("finished double sampling")
        elif self.flag == 1:
            #resampling
            if self.training_method == "LP":
                lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
                if self.use_verifai:
                    self.feedback_result = -lp
                self.resampling_weights[self.working_index] = lp
                self.buffer_last_reward[self.working_index] = total_reward
            elif self.training_method == "EL":
                inverse_reward = 1 / (self.steps_taken + 100)
                if(self.steps_taken <= 50):
                    inverse_reward = 0
                self.resampling_weights[self.working_index] = inverse_reward
            else:
                print("BIG ISSUE, resmaled but not special PLR")
        else:
            if self.training_method == "LP":
                self.buffer_last_reward = np.append(self.buffer_last_reward, total_reward)
            elif self.training_method == "EL":
                inverse_reward = 1 / (self.steps_taken + 100)
                if(self.steps_taken <= 50):
                    inverse_reward = 0
                self.resampling_weights = np.append(self.resampling_weights, inverse_reward)