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

    # Number of particles
    number_of_particles = 5

    # Weights
    # normalized_weights = number_of_particles * [1.0 / number_of_particles]
    unnormalized_weights = np.random.random_sample((number_of_particles,))
    normalized_weights = [w / np.sum(unnormalized_weights) for w in unnormalized_weights]

    # States
    particle_states = range(1, number_of_particles+1)

    # Weighted samples
    weighted_particles = zip(normalized_weights, particle_states)

    # Resampling algorithms that must be compared
    resampler = Resampler()
    methods = [ResamplingAlgorithms.MULTINOMIAL,
               ResamplingAlgorithms.RESIDUAL,
               ResamplingAlgorithms.STRATIFIED,
               ResamplingAlgorithms.SYSTEMATIC]

    # Number of resample steps
    num_steps = 100000

    print('Input samples: {}'.format(weighted_particles))

    # Compare resampling algorithms
    for method in methods:
        print('Testing {} resampling'.format(method.name))
        all_results = []
        for i in range(0, num_steps+1):
            # Resample
            weighted_samples = resampler.resample(weighted_particles,
                                                  number_of_particles,
                                                  method)

            # Store number of occurrences each particle in array
            sampled_states = [state[1] for state in weighted_samples]
            all_results.append([sampled_states.count(state_i) for state_i in particle_states])

        # Print results for current resampling algorithm
        print('mean #occurrences: {}'.format(np.mean(all_results, axis=0)))
        print('std  #occurrences: {}'.format(np.std(all_results, axis=0)))
