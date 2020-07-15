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


def binary_search(cumulativeList, lower, upper, x):
    """
    Binary search: find the index i for which cumulativeList[i-1] < x <= cumulativeList[i] within
    cumulativeList[lower:upper]. Returns -1 if failed.
    :param cumulativeList: List of elements that increase with increasing index.
    :param lower: Lower bound index
    :param upper: Upper bound index
    :param x: value for which has to be checked
    :return: Index, -1 if invalid input provided or failure.
    """

    # Validate input data
    if upper > lower and len(cumulativeList) > 0:

        # Find middle point index
        mid = int(round(lower + (upper - lower) / 2.0))

        # If middle element is first element above x (or equal)
        if cumulativeList[mid - 1] < x <= cumulativeList[mid]:
            return mid
        elif cumulativeList[mid] > x:
            # Middle above x, but not first element above x: must be in left subarray
            return binary_search(cumulativeList, lower, mid - 1, x)
        else:
            # Middle below x, must be in right subarray
            return binary_search(cumulativeList, mid + 1, upper, x)
    elif upper is lower and len(cumulativeList) > 0:
        # Trivial case
        return upper
    else:
        # Invalidate input
        return -1


def compute_threshold(weights, N):
    """
    The threshold ct can be computed by solving:
      N = sum_M min(wm / ct, 1.0)              (1)

    There are 3 possible cases:
    1) M = N
       In this case setting ct = min(wm) renders all wm / ct >= 1.0, hence all terms inside the summation will be 1.0 due
       to the minumum that will be taken. This solution is NOT desirable in the use case of optimal core since the
       output samples will always be a copy of the input samples (meaning no real core occurs). Instead of picking
       this trivial solution, we skip to case 3.

    2) N > M
       In this case we start by setting ct1 = max(wm), such that min(wm/ct1, 1.0) = wm/ct1 for all weights. Evaluating
       the sum in this case leads to sum_M (wm/ct1) = sum <= M < N.
       Finding ct can be obtained by first solving:
         N = 1 / ct2 * sum -> ct2 = sum / N
       The overall solution ct = ct1*ct2 which is the exact solution of (1). This strategy will also be used in case 1
       (M = N).

    3) M > N
       In this case min(wm) < ct <= max(wm). The exact solution ct will in general not be equal to one of the particle
       weights, however, the threshold will be used as weight threshold, hence we only have to make sure that
         weight_m >= ct_approx
       and
         weight_m >= ct_exact
       give the same result for the given list of weights. The approximate solution ct_approx will be equal to the
       largest  weight that satisfies weight_m <= ct_optimal and can be found by trying all weights in descending order
       and stop trying once the right hand side (1) exceeds the left hand side.
    :param weights: given list of weights
    :param N: desired number of particles
    :return: threshold ct
    """

    # Number of samples
    M = len(weights)

    # Sort weights in descending order
    weights_sorted = sorted(weights, reverse=True)

    # # Case 1: Not desirable to pick trivial solution, skip to case 3
    # if N == M:
    #     print("Number of particles unchanged, ct is {}".format(weights_sorted[M - 1]))
    #     return weights_sorted[M - 1]

    # Case 2
    # @todo: check if this solution works in practice
    if N > M:
        # Get maximum weight, this value is used for ct1
        ct1 = weights_sorted[0]

        # Compute summation using the maximum weight as ct1
        summation = sum([w / ct1 for w in weights_sorted])

        # ct is product of ct1 and ct2, where ct2 = summation / N and that ensures N = summation / ct2
        return summation / N * ct1

    # Case 3
    current_index = -1  # index starts at 0, we start with adding 1, hence initialize at -1
    summation = 0       # initialize for starting while loop
    while summation < N:
        # Update index
        current_index += 1

        # This cannot happen (last weight (is minimum weight) would lead to summation being at least M, and M >= N)
        assert (current_index != M)

        # Get current threshold
        ct = weights_sorted[current_index]

        # Recompute sum
        summation = sum([min(wm / ct, 1.0) for wm in weights_sorted])

    # Should not be possible to have current_index being not positive at this point
    assert (current_index > 0)

    # While loop breaks if summation exceeds N, ct_approx equals last weight that led to summation not exceeding N
    ct_approx = weights_sorted[current_index - 1]

    return ct_approx


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

    # Return index of first sample for which cumulative sum is above u (using binary search)
    return binary_search(Q, 0, len(Q) - 1, u)


def compute_required_number_of_particles_KLD(k, epsilon, upper_quantile):
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
    x = 1.0 - 2.0 / (9.0*(k-1)) + np.sqrt(2.0 / (9.0*(k-1)) * upper_quantile)
    return np.ceil((k-1) / (2.0*epsilon) * x * x * x)
