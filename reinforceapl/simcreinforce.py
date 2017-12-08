# -*- coding: utf-8 -*-
"""
Reinforcement algorithm for simc RL.

@author: skasch
"""

from collections import defaultdict
from math import floor, ceil

import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier as DTC


class Policy:
    """
    Defines a policy.
    """

    def __init__(self, discount=0.95, max_depth=None):
        self.discount = discount
        self.value_dict = defaultdict(lambda: [])
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(float)
        self.param_ranges = defaultdict(lambda: (float('Inf'), 0.))
        self.visited_states = []
        self.valid_params = []
        self.valid_actions = []
        self.update_ranges = True
        self.action_models = {}
        self.tree_model = DTC(max_depth=max_depth)

    def learn_from_episode(self, episode):
        """
        Updates the value dict from the episode.
        """
        for action, state, _ in episode:
            if self.update_ranges:
                for param, score in state.items():
                    range_min, range_max = self.param_ranges[param]
                    self.param_ranges[param] = (floor(min(range_min, score)),
                                                ceil(max(range_max, score)))
                self.valid_params = sorted([
                    param for param, range_ in self.param_ranges.items()
                    if range_[1] > range_[0]])
            sa_pair = (frozenset(state.items()), action)
            self.visited_states.append(sa_pair[0])
            # Find the first occurance of the (state, action) pair in the episode
            first_idx = next(i for i, x in enumerate(episode)
                             if x[0] == action and x[1] == state)
            # Sum up all rewards since the first occurance
            value = sum(x[2] * (self.discount**i)
                        for i, x in enumerate(episode[first_idx:]))
            # Calculate average return for this state over all sampled episodes
            self.returns_sum[sa_pair] += value
            self.returns_count[sa_pair] += 1.0
            value_state = {'value': (self.returns_sum[sa_pair]
                                     / self.returns_count[sa_pair]),
                           'state': state}
            self.value_dict[action].append(value_state)
        self.valid_actions = sorted(self.value_dict.keys())

    def normalize_state(self, state):
        ranges = self.param_ranges
        return [((dict(state).get(k, ranges[k][0]) - ranges[k][0])
                 / (ranges[k][1] - ranges[k][0])) for k in self.valid_params]

    def update_model(self, action):
        values = []
        states = []
        for state_value in self.value_dict[action]:
            values.append(state_value['value'])
            states.append(self.normalize_state(state_value['state']))
        clf = SVR(C=0.2, epsilon=0.2)
        clf.fit(states, values)
        self.action_models[action] = clf
    
    def update_action_models(self):
        for action in self.valid_actions:
            self.update_model(action)

    def update_decision_tree(self):
        norm_states = []
        best_actions = []
        for state in self.visited_states:
            norm_state = self.normalize_state(state)
            norm_states.append(norm_state)
            best_value = -1
            best_action = ''
            for action in self.valid_actions:
                this_value = self.action_models[action].predict(
                    np.array(norm_state).reshape(1, -1))
                if this_value > best_value:
                    best_value = this_value
                    best_action = action
            best_actions.append(best_action)
        self.tree_model.fit(norm_states, best_actions)


class Teacher:
    """
    Defines the teacher.
    """

    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.done = False

    def new_episode(self):
        """
        Returns a new episode from the environment.
        """
        if self.done:
            self.env.reset()
        self.done = True
        episode = []
        while True:
            step = self.env.step()
            print(step)
            episode.append((step[0], step[1], step[2]))
            if step[3]:
                break
        return episode
