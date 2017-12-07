# -*- coding: utf-8 -*-
"""
Build a simc-backed environment for Reinforcement Learning.

@author: skasch
"""

import platform
import subprocess
import os

from .resultparser import SimcResult

class SimcEnv:
    """
    simc environment for RL.
    """

    def __init__(self, simc_folder, profile_path, json_path):
        self.simc_folder = simc_folder
        self.profile_path = profile_path
        self.json_path = json_path
        self.call_simc()
        self.simc_result = SimcResult(self.json_path)
        self.current_step = 0
        self.results = self.result_generator()

    def simc_binary(self):
        """
        Return the name of the simc binary.
        """
        ext = '.exe' if platform.system() == 'Windows' else ''
        return f'simc{ext}'

    def simc_command(self):
        """
        Return the full command to call simc.
        """
        return os.path.join(self.simc_folder, self.simc_binary())

    def simc_args(self):
        """
        Return the arguments to call simc with.
        """
        args = []
        args.append(f'json2={self.json_path}')
        args.append('json_full_states=1')
        args.append('iterations=1')
        args.append(self.profile_path)
        return args

    def call_simc(self):
        """
        Call simc with current arguments.
        """
        simc_process = subprocess.Popen(
            [self.simc_command()] + self.simc_args())
        simc_process.wait()

    def reset(self):
        """
        BuiLd the result of the next simulation/episode.
        """
        self.call_simc()
        self.simc_result = SimcResult(self.json_path)
        self.current_step = 0
        self.results = self.result_generator()

    def result_generator(self):
        """
        Generate the next result.
        """
        while self.current_step < len(self.simc_result):
            yield (self.simc_result.reward(self.current_step),
                   self.simc_result.state(self.current_step),
                   self.current_step >= len(self.simc_result) - 1)
            self.current_step += 1

    def step(self):
        """
        Returns the next step.
        """
        try:
            return next(self.results)
        except StopIteration:
            return None
