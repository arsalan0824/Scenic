from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable
from collections import deque
import numpy as np

# A custom exception to handle resets within the generator loop
import gymnasium as gym
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

        self.last_10_episode_coverages = deque(maxlen=10)
        self.total_episodes_completed = 0
        self.current_total_coverage_sum = 0

    def _make_run_loop(self):
        while True:
            try:
                # The feedback_result (new_clip_range) is passed to scenario generation
                scene, _ = self.scenario.generate(feedback=self.feedback_result)
                #make a variable so self.current_total_coverage_sum is accessible in simulator.py
                
                with self.simulator.simulateStepped(scene, maxSteps=self.max_steps) as simulation:
                    simulation.current_total_coverage_sum = self.current_total_coverage_sum
                    steps_taken = 0
                    done_episode = lambda: not (simulation.result is None) or (simulation.get_truncation())
                    truncated_episode = lambda: (steps_taken >= self.max_steps)

                    observation = simulation.get_obs()
                    initial_info = {}
                    actions = yield observation, initial_info

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