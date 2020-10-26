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
    """
    In this program a particle set will be generated and different resampling algorithms will be used to resample (with 
    replacement) from this particle set. The probability of selecting a particle is proportional to its weight.
    
    In the end the number of times each particle has been selected (this should be the same for all resampling 
    algorithms) is printed. Furthermore, the standard deviations are shown (the variance in the number of times each 
    particle has been selected. This is a measure of the predictability of the resampling algorithm. In other words, the
    diversity in sample set when performing resampling multiple times on exactly the same particle set.
    """

    print("Compare four resampling algorithms.")

    # Number of particles
    number_of_particles = 5

    # Determine weights (random weights), normalize such that they sum up to one.
    unnormalized_weights = np.random.random_sample((number_of_particles,))
    normalized_weights = [w / np.sum(unnormalized_weights) for w in unnormalized_weights]

    # States range from 1 to the number of particles
    particle_states = range(1, number_of_particles+1)

    # Weighted samples
    weighted_particles = list(zip(normalized_weights, particle_states))

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
