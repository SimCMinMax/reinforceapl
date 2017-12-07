# -*- coding: utf-8 -*-
"""
Parse json results from simc.

@author: skasch
"""

import json
import logging

import numpy as np

class SimcResult:
    """
    The result of a simc simulation.
    """

    def __init__(self, json_path):
        self.path = json_path
        with open(json_path) as json_file:
            self.data = json.load(json_file)

        print(f'Extracting result: simc version {self.data["version"]}, build '
              f'time {self.data["build_date"]} {self.data["build_time"]}')
        if self.data['ptr_enabled']:
            logging.warning('The simulation has PTR enabled.')
        if self.data['beta_enabled']:
            logging.warning('The simulation has beta enabled.')

        self.sim = self.data['sim']
        self.main_player = self.sim['players'][0]
        self.collected_data = self.main_player['collected_data']
        self.action_sequence = self.collected_data['action_sequence']
        self.rewards = []
        self.resource_timelines = {}

        self.build_rewards()
        self.build_resource_timelines()

    def build_rewards(self):
        """
        Build the rewards array (damage done between two steps).
        """
        self.rewards = ([self.action_sequence[0]['dmg']]
                        + np.diff([a['dmg'] for a in self.action_sequence]))

    def build_resource_timelines(self):
        """
        Build the resources timelines.
        """
        self.resource_timelines = {
            k: v['data'] for (k, v) in
            self.collected_data['resource_timelines'].items()}

    def __len__(self):
        return len(self.rewards)

    def reward(self, idx):
        """
        Return the idx-th reward.
        """
        return self.rewards[idx]

    def state(self, idx):
        """
        Return the idx-th state.
        """
        return self.action_sequence[idx]
