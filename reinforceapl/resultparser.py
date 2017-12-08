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

    def action(self, idx):
        """
        Return the idx-th action.
        """
        try:
            return self.action_sequence[idx]['name']
        except KeyError:
            return 'wait'

    def state(self, idx):
        """
        Return the idx-th state.
        """
        idx_state = {}
        idx_state['time'] = self.action_sequence[idx]['time']
        for buff in self.action_sequence[idx].get('buffs', []):
            idx_state.update({f'buff.{buff["name"]}.stacks': buff['stacks'],
                              f'buff.{buff["name"]}.remains': buff['remains']})
        for cdn in self.action_sequence[idx].get('cooldowns', []):
            idx_state.update({f'cd.{cdn["name"]}.stacks': cdn['stacks'],
                              f'cd.{cdn["name"]}.remains': cdn['remains']})
        for target in self.action_sequence[idx].get('targets', []):
            tgt = target['name']
            for debuff in target.get('debuffs', []):
                dbf = debuff['name']
                idx_state.update(
                    {f'debuff.{tgt}.{dbf}.stacks': debuff['stack'],
                     f'debuff.{tgt}.{dbf}.remains': debuff['remains']})
        return idx_state
