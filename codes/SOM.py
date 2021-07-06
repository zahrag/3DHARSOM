
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from numpy import linalg as LA


class SOM:

    def __init__(self, learning, outputsize_x, outputsize_y, inputsize, sigma, softmax_exponent, max_epoch):

        self.name = 'SOM'
        self.learning = learning
        self.outputsize_x = outputsize_x
        self.outputsize_y = outputsize_y
        self.inputsize = inputsize
        self.sigma = sigma
        self.softmax_exponent = softmax_exponent
        self.max_epoch = max_epoch
        self.metric = 'Euclidean'
        self.normalize_input = False
        self.normalize_weights = False
        self.softmax_normalization = True
        self.neighborhood_decay = 0.9999
        self.neighborhood_min = 1
        self.learningRate = 0.1
        self.learningRate_decay = 0.9999
        self.learningRate_min = 0.01
        self.neighborhood_radius = outputsize_x
        self.node_map = np.zeros((outputsize_x, outputsize_y, 2))
        self.weights = np.random.rand(outputsize_x, outputsize_y, inputsize)  # Rows, Columns, Depth

        for i in range(outputsize_x):
            for j in range(outputsize_y):
                self.node_map[i, j, 0] = i
                self.node_map[i, j, 1] = j

    def normalize(self, state):

        if self.normalize_input:
            state /= LA.norm(np.expand_dims(state, axis=0))

        return state

    def soft_max_normalization(self, state):

        m = np.max(state)
        if m != 0:
            state /= m

        return state

    def set_activity(self, state):

        if self.metric == 'Euclidean':
            dist = np.sum((state - self.weights) ** 2, axis=2)
            activity = np.exp(-dist / self.sigma)

        else:
            # Scalar Product
            mat_mul = state * self.weights
            activity = mat_mul.sum(axis=2)

        if self.softmax_exponent != 1:
            activity = activity ** self.softmax_exponent

        if self.softmax_normalization:
            activity = self.soft_max_normalization(activity)

        return activity

    def find_winning_node(self, activity):

        winner_x, winner_y = np.unravel_index(np.argmax(activity, axis=None), activity.shape)
        winning_node = np.array([winner_x, winner_y])

        return winning_node

    def learn(self, state, winner):

        dis = np.sum((self.node_map - winner) ** 2, axis=2)
        gus = np.exp(-dis / (2 * self.neighborhood_radius ** 2))
        err = state - self.weights
        self.weights += self.learningRate * (err.T * gus.T).T

    def learning_decay(self):

        self.learningRate *= self.learningRate_decay
        if self.learningRate < self.learningRate_min:
            self.learningRate = self.learningRate_min

        self.neighborhood_radius *= self.neighborhood_decay
        if self.neighborhood_radius < self.neighborhood_min:
            self.neighborhood_radius = self.neighborhood_min

    def run_SOM(self, state):

        state = self.normalize(state)

        activity = self.set_activity(state)

        winner = self.find_winning_node(activity)

        if self.learning:
            self.learn(state, winner)
            self.learning_decay()

        return activity, winner




