#!/usr/bin/env python

import numpy as np


def cumulative_sum(weights):
    """
    Compute cumulative sum of a list of scalar weights

    :param weights: list with weights
    :return: list containing cumulative weights, length equal to length input
    """
    return np.cumsum(weights).tolist()


def replication(samples):
    """
    Deterministically replicate samples.

    :param samples: A list in which each element again is a list containing the sample
    and the number of times the sample needs to be replicated, e.g.:
      samples = [(x1, 2), (x2, 0), ..., (xM, 1)]
    :return: list of replicated samples: [x1, x1, ..., xM]
    """

    # A perhaps more understandable way to solve this could be:
    # replicated_samples = []

    # for m in range(1, len(samples)+1):
    #     xk, Nk = samples[m-1]
    #     for unused in range(1, Nk+1):
    #         replicated_samples.append(xk)
    # return replicated_samples

    # Same result: repeat each state (denoted by s[0]) Nk times (denoted by s[1])
    return [l for s in samples for l in s[1] * [s[0]]]


def naive_search(cumulative_list, x):
    """
    Find the index i for which cumulativeList[i-1] < x <= cumulativeList[i] within cumulativeList[lower:upper].

    :param cumulative_list: List of elements that increase with increasing index.
    :param x: value for which has to be checked
    :return: Index
    """

    m = 0
    while cumulative_list[m] < x:
        m += 1
    return m


def add_weights_to_samples(weights, unweighted_samples):
    """
    Combine weights and unweighted samples into a list of lists:
    [[w1, [particle_state]], [w2, [particle_state2]], ...]
    :param weights: Sample weights
    :param unweighted_samples: Sample states
    :return: list of lists
    """
    weighted_samples = [list(ws) for ws in zip(weights, unweighted_samples)]
    return weighted_samples


def generate_sample_index(weighted_samples):
    """
    Sample a particle from the discrete distribution consisting out of all particle weights.

    :param weighted_samples: List of weighted particles
    :return: Sampled particle index
    """

    # Check input
    if len(weighted_samples) < 1:
        print("Cannot sample from empty set")
        return -1

    # Get list with only weights
    weights = [weighted_sample[0] for weighted_sample in weighted_samples]

    # Compute cumulative sum for all weights
    Q = cumulative_sum(weights)

    # Draw a random sample u in [0, sum_all_weights]
    u = np.random.uniform(1e-6, Q[-1], 1)[0]

    # Return index of first sample for which cumulative sum is above u
    return naive_search(Q, u)


def compute_required_number_of_particles_kld(k, epsilon, upper_quantile):
    """
    Compute the number of samples needed within a particle filter when k bins in the multidimensional histogram contain
    samples. Use Wilson-Hilferty transformation to approximate the quantiles of the chi-squared distribution as proposed
    by Fox (2003).

    :param epsilon: Maxmimum allowed distance (error) between true and estimated distribution.
    :param upper_quantile: Upper standard normal distribution quantile for (1-delta) where delta is the probability that
    the error on the estimated distribution will be less than epsilon.
    :param k: Number of bins containing samples.
    :return: Number of required particles.
    """
    # Helper variable (part between curly brackets in (7) in Fox paper
    x = 1.0 - 2.0 / (9.0*(k-1)) + np.sqrt(2.0 / (9.0*(k-1))) * upper_quantile
    return np.ceil((k-1) / (2.0*epsilon) * x * x * x)
