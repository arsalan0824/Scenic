from scenic.core.simulators import Simulator, Simulation, TerminationType
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable

import random
import numpy as np

#TODO make ResetException
class ResetException(Exception):
    def __init__(self):
        super().__init__("Resetting")

class ScenicGymEnv(gym.Env):
    """
    verifai_sampler now not an argument added in here, but one specified int he Scenic program
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4} # TODO placeholder, add simulator-specific entries
    
    def __init__(self, 
                 scenario : Scenario,
                 simulator : Simulator,
                 render_mode=None, 
                 max_steps = 1000,
                 observation_space : spaces.Dict = spaces.Dict(),
                 action_space : spaces.Dict = spaces.Dict(),
                 record_scenic_sim_results : bool = True,
                 feedback_fn : callable = lambda x: x): # empty string means just pure scenic???

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.max_steps = max_steps - 1 # FIXME, what was this about again?
        self.simulator = simulator
        self.scenario = scenario
        self.simulation_results = []

        self.feedback_result = None
        self.loop = None
        self.record_scenic_sim_results = record_scenic_sim_results
        self.feedback_fn = feedback_fn
        
        #CHANGEABLE STUFF, CHECK BEFORE EACH RUN ALSO CHECK THAT THE BUFFER IS EMPTY
        self.training_method = "LP"
        self.use_plr = False
        self.use_verifai = True
        #Random: default
        #LP: Prioritized level replay based off learning potential
        #GAE: Generalized advantage estimation NOTE I HAVEN"T ADDED VEIRFAI + GAE YET
        self.is_continuation = False # are you reloading the buffer form the disk
        self.buffer_p = 0.5 # probability of resampling
        
        #load arrays
        self.buffer_filenames = np.load("../../../../../../buffer/buffer_filenames.npy") if self.is_continuation else np.array([]) # filenames of the scenes in the buffer its redundant and will delete if have extra time or code feels too messy
        self.buffer_learning_potential =  np.load("../../../../../../buffer/buffer_learning_potential.npy") if self.is_continuation else np.array([]) # learning potential of the scenes in the buffer note that for easy genrealization, this is juts the prioritiy weights, whether it be LP or GAE
        self.buffer_last_reward =  np.load("../../../../../../buffer/buffer_last_reward.npy") if self.is_continuation else np.array([]) # last reward of the scenes in the buffer
        if(self.is_continuation):
            print(f"loaded files: {self.buffer_filenames}, learning potential: {self.buffer_learning_potential}, last reward: {self.buffer_last_reward}")
        
        #extra variables for the run loop
        self.working_index = -1
        self.flag = 0
        self.counting_reward = 0
        
        if self.use_plr and self.training_method not in ("LP", "GAE"):
            raise ValueError(
                f"use_plr=True but training_method={self.training_method!r}. "
                "Must be one of 'LP' or 'GAE' if use_plr is enabled."
            )

    def _make_run_loop(self):
        while True:
            try:
                scene = None
                self.is_resampling = random.uniform(0, 1) < self.buffer_p
                #sample or resample scenes
                if not self.use_plr and self.training_method != "GAE" and self.training_method != "LP":
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                elif self.training_method == "LP" and len(self.buffer_learning_potential) != len(self.buffer_last_reward):
                    # finish sampling doubled scene
                    self.flag = 0
                    self.working_index = len(self.buffer_filenames) - 1
                    with open(f"../../../../../../buffer/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print("Double sampling")
                elif self.use_plr and self.is_resampling and len(self.buffer_filenames) > 0:
                    # resample from buffer
                    self.flag = 1
                    prob_distribution = self.buffer_learning_potential / np.sum(self.buffer_learning_potential)
                    self.working_index = np.random.choice(len(self.buffer_filenames), p=prob_distribution)
                    with open(f"../../../../../../buffer/scene_{self.working_index}.bin", "rb") as f:
                        scene = self.scenario.sceneFromBytes(f.read())
                    print(f"Resampling from buffer with index {self.working_index}")
                else:
                    # sample new scene
                    self.flag = 2
                    scene, _ = self.scenario.generate(feedback=self.feedback_result)
                    with open(f"../../../../../../buffer/scene_{len(self.buffer_filenames)}.bin", "wb") as f:
                        f.write(self.scenario.sceneToBytes(scene=scene))
                    self.buffer_filenames = np.append(self.buffer_filenames, len(self.buffer_filenames))
                    self.working_index = len(self.buffer_filenames) - 1
                    print(f"Sampling new scene with index {self.working_index}")
                
                self.counting_reward = 0
                with self.simulator.simulateStepped(scene, maxSteps=self.max_steps) as simulation:
                    steps_taken = 0
                    # this first block before the while loop is for the first reset call
                    done = lambda: not (simulation.result is None) or (simulation.get_truncation()) # allows for early truncation
                    truncated = lambda: (steps_taken >= self.max_steps)  # TODO handle cases where it is done right on maxsteps
                    observation = simulation.get_obs()
                    info = simulation.get_info() 
                    actions = yield observation, info
                    simulation.actions = actions # TODO add action dict to simulation interfaces
                    while not done():
                        # Probably good that we advance first before any action is set.
                        # this is consistent with how reset works
                        simulation.advance()
                        steps_taken += 1
                        observation = simulation.get_obs()
                        info = simulation.get_info()
                        reward = simulation.get_reward()
                        self.counting_reward += reward
                        if done():
                            if simulation.result is None:
                                simulation.terminateSimulation(TerminationType.terminatedByUser, "early truncation")
                            print("Simulation done")
                            self.logScores()
                            if not self.use_verifai:
                                self.feedback_result = self.feedback_fn(simulation.result)
                            if self.record_scenic_sim_results:
                                self.simulation_results.append(simulation.result)
                            # simulation.destroy() # FIXME...might redundant?
                            actions = yield observation, reward, done(), truncated(), info
                            
                            break # a little unclean right here

                        actions = yield observation, reward, done(), done(), info
                        simulation.actions = actions # TODO add action dict to simulation interfaces
                    print("Exitedwhile loop")
            except ResetException:
                print("reset exception caught")
                continue

    def reset(self, seed=None, options=None): # TODO will setting seed here conflict with VerifAI's setting of seed?
        # only setting enviornment seed, not torch seed?
        super().reset(seed=seed)
        if self.loop is None:
            print("self loop doesnt exist, creating new one")
            self.loop = self._make_run_loop()
            observation, info = next(self.loop) # not doing self.scene.send(action) just yet
        else:
            observation, info = self.loop.throw(ResetException())


        return observation, info
        
    def step(self, action):
        assert not (self.loop is None), "self.loop is None, have you called reset()?"

        observation, reward, terminated, truncated, info = self.loop.send(action)
        return observation, reward, terminated, truncated, info

    def render(self): # TODO figure out if this function has to be implemented here or if super() has default implementation
        """
        likely just going to be something like simulation.render() or something
        """
        # FIXME for one project only...also a bit hacky...
        # self.env.render()
        pass

    def close(self):
        self.simulator.destroy()
        
    def logScores(self):
        total_reward = self.counting_reward
        if(total_reward == 0):
            print("TOTAL REWRAD is 0! suspiciosu!")
        #log rewards and learning potential
        if self.flag == 0:
            #double sampling
            lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
            if self.use_verifai:
                self.feedback_result = -lp
            self.buffer_learning_potential = np.append(self.buffer_learning_potential, lp)
            self.buffer_last_reward[self.working_index] = total_reward
            print("finished double sampling")
        elif self.flag == 1:
            #resampling
            if(self.working_index >= len(self.buffer_last_reward)):
                print(f"Warning: working index {self.working_index} is out of bounds for buffer_last_reward with length {len(self.buffer_last_reward)}")
            if self.training_method == "LP":
                lp = abs(total_reward - self.buffer_last_reward[self.working_index]) + 1e-8
                if self.use_verifai:
                    self.feedback_result = -lp
                self.buffer_learning_potential[self.working_index] = lp
                self.buffer_last_reward[self.working_index] = total_reward
            elif self.training_method == "GAE":
                lp = total_reward
                self.buffer_learning_potential[self.working_index] = lp
            else:
                print("BIG ISSUE, resmaled but not special PLR")
        else:
            if self.training_method == "LP":
                self.buffer_last_reward = np.append(self.buffer_last_reward, total_reward)
            elif self.training_method == "GAE":
                lp = total_reward
                self.buffer_learning_potential = np.append(self.buffer_learning_potential, lp)
            else:
                print("BIG ISSUE, resampled but not special PLR")
            print("appened last reward")
        print(self.buffer_learning_potential)
        np.save("../../../../../../buffer/buffer_filenames.npy", self.buffer_filenames)
        np.save("../../../../../../buffer/buffer_learning_potential.npy", self.buffer_learning_potential)
        np.save("../../../../../../buffer/buffer_last_reward.npy", self.buffer_last_reward)
        print("Saved buffer data to disk")