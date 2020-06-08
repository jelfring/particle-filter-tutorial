#!/usr/bin/env python

# Supported resampling methods
from core.resampling import Resampler, ResamplingAlgorithms

import numpy as np


def get_states(weighted_samples):
    """
    Get the states from set of weighted particles.

    :param weighted_samples: list of weighted particles
    :return: list of particle states without weights
    """
    return [ws[1] for ws in weighted_samples]


if __name__ == '__main__':

    print("Compare four resampling algorithms.")

    # Artificial particle set
    weights = [0.01, 0.25, 0.04, 0.1, 0.05, 0.05, 0.25, 0.11, 0.12, 0.03]
    number_of_particles = len(weights)
    particle_states = range(1, number_of_particles+1)
    weighted_particles = zip(weights, particle_states)

    resampler = Resampler()
    methods = [ResamplingAlgorithms.MULTINOMIAL,
               ResamplingAlgorithms.RESIDUAL,
               ResamplingAlgorithms.STRATIFIED,
               ResamplingAlgorithms.SYSTEMATIC]

    print('Input: {}'.format(weighted_particles))

    # Compare resampling algorithms
    for method in methods:
        print('Testing {} resampling'.format(method.name))
        for i in range(1000):
            samples = resampler.resample(weighted_particles,
                                         number_of_particles,
                                         method)
        # Count occurrences
        result = np.array([np.array(xi) for xi in [samples.count(state) for state in particle_states]])
        print('mean number of occurrences: {}'.format(np.mean(result, axis=0)))
