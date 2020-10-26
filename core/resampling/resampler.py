#!/usr/bin/env python

# Numpy
import numpy as np

# Enum
from enum import Enum

# Deep copy samples
import copy

# Helper functions
from .resampling_helpers import *


class ResamplingAlgorithms(Enum):
    MULTINOMIAL = 1
    RESIDUAL = 2
    STRATIFIED = 3
    SYSTEMATIC = 4


class Resampler:
    """
    Resample class that implements different resampling methods.
    """

    def __init__(self):
        self.initialized = True

    def resample(self, samples, N, algorithm):
        """
        Resampling interface, perform resampling using specified method

        :param samples: List of (weight, sample)-lists that need to be resampled
        :param N: Number of samples that must be resampled.
        :param algorithm: Preferred method used for resampling.
        :return: List of weighted samples.
        """

        if algorithm is ResamplingAlgorithms.MULTINOMIAL:
            return self.__multinomial(samples, N)
        elif algorithm is ResamplingAlgorithms.RESIDUAL:
            return self.__residual(samples, N)
        elif algorithm is ResamplingAlgorithms.STRATIFIED:
            return self.__stratified(samples, N)
        elif algorithm is ResamplingAlgorithms.SYSTEMATIC:
            return self.__systematic(samples, N)

        print("Resampling method {} is not specified!".format(algorithm))

    @staticmethod
    def __multinomial(samples, N):
        """
        Particles are sampled with replacement proportional to their weight and in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be
        resampled between 0 and N times.

        Computational complexity: O(N log(M)

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """

        # Get list with only weights
        weights = [weighted_sample[0] for weighted_sample in samples]

        # Compute cumulative sum
        Q = cumulative_sum(weights)

        # As long as the number of new samples is insufficient
        n = 0
        new_samples = []
        while n < N:

            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Naive search (alternative: binary search)
            m = naive_search(Q, u)

            # Add copy of the state sample (uniform weights)
            new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

            # Added another sample
            n += 1

        return new_samples

    def __residual(self, samples, N):
        """
        Particles should at least be present floor(wi/N) times due to first deterministic loop. First Nt new samples are
        always the same (when running the function with the same input multiple times).

        Computational complexity: O(M) + O(N-Nt), where Nt is number of samples in first deterministic loop

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """

        # Number of incoming samples
        M = len(samples)

        # Initialize
        weight_adjusted_samples = []
        replication_samples = []
        for m in range(0, M):
            # Copy sample and ease of writing
            wm, xm = copy.deepcopy(samples[m])

            # Compute replication
            Nm = np.floor(N * wm)

            # Store weight adjusted sample (and avoid division of integers)
            weight_adjusted_samples.append([wm - float(Nm) / N, xm])

            # Store sample to be used for replication
            replication_samples.append([xm, int(Nm)])

        # Replicate samples
        new_samples_deterministic = replication(replication_samples)
        Nt = len(new_samples_deterministic)

        # Normalize new weights if needed
        if N != Nt:
            for m in range(0, M):
                weight_adjusted_samples[m][0] *= float(N) / (N - Nt)

        # Resample remaining samples (__multinomial return weighted samples, discard weights)
        new_samples_stochastic = [s[1] for s in self.__multinomial(weight_adjusted_samples, N - Nt)]

        # Return new samples
        weighted_new_samples = add_weights_to_samples(N * [1.0/N], new_samples_deterministic + new_samples_stochastic)
        return weighted_new_samples

    @staticmethod
    def __stratified(samples, N):
        """
        Loop over cumulative sum once hence particles should keep same order (however some disappear, others are
        replicated).

        Computational complexity: O(N)

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """

        # Compute cumulative sum on normalized weights
        weights = [weighted_sample[0] for weighted_sample in samples]
        normalized_weights = [w / sum(weights) for w in weights]
        Q = cumulative_sum(normalized_weights)

        # As long as the number of new samples is insufficient
        n = 0
        m = 0  # index first element
        new_samples = []
        while n < N:

            # Draw a random sample u0 and compute u
            u0 = np.random.uniform(1e-10, 1.0 / N, 1)[0]
            u = u0 + float(n) / N

            # u increases every loop hence we only move from left to right (once) while iterating Q

            # Get first sample for which cumulative sum is above u
            while Q[m] < u:
                m += 1  # no need to reset m, u always increases

            # Add state sample (weight, state)
            new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

            # Added another sample
            n += 1

        # Return new samples
        return new_samples

    @staticmethod
    def __systematic(samples, N):
        """
        Loop over cumulative sum once hence particles should keep same order (however some disappear, other are
        replicated). Variance on number of times a particle will be selected lower than with stratified resampling.

        Computational complexity: O(N)

        :param samples: Samples that must be resampled.
        :param N: Number of samples that must be generated.
        :return: Resampled weighted particles.
        """
        # Compute cumulative sum
        weights = [weighted_sample[0] for weighted_sample in samples]
        Q = cumulative_sum(weights)

        # Only draw one sample
        u0 = np.random.uniform(1e-10, 1.0 / N, 1)[0]

        # As long as the number of new samples is insufficient
        n = 0
        m = 0  # index first element
        new_samples = []
        while n < N:

            # Compute u for current particle (deterministic given u0)
            u = u0 + float(n) / N

            # u increases every loop hence we only move from left to right while iterating Q

            # Get first sample for which cumulative sum is above u
            while Q[m] < u:
                m += 1

            # Add state sample (uniform weights)
            new_samples.append([1.0/N, copy.deepcopy(samples[m][1])])

            # Added another sample
            n += 1

        # Return new samples
        return new_samples
