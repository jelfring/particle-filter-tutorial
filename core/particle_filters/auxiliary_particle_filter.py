from .particle_filter_base import ParticleFilter
from core.resampling import naive_search, cumulative_sum

import numpy as np


class AuxiliaryParticleFilter(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
    """

    @staticmethod
    def sample_multinomial_indices(samples):
        """
        Particles indices are sampled with replacement proportional to their weight an in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be resampled
        between 0 and N times.
        Computational complexity: O(N log(M)

        :param samples: Samples that must be resampled.
        :return: Resampled indices.
        """

        # Number of samples
        N = len(samples)

        # Get list with only weights
        weights = [weighted_sample[0] for weighted_sample in samples]

        # Compute cumulative sum
        Q = cumulative_sum(weights)

        # As long as the number of new samples is insufficient
        n = 0
        new_indices = []
        while n < N:
            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Get first sample for which cumulative sum is above u using naive search
            m = naive_search(Q, u)

            # Store index
            new_indices.append(m)

            # Added another sample
            n += 1

        return new_indices

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and adopting the auxiliary sampling importance
        core (ASIR) scheme explained in [1].

        [1] M. S. Arulampalam, S. Maskell, N. Gordon and T. Clapp, "A tutorial on particle filters for online
        nonlinear/non-Gaussian Bayesian tracking," in IEEE Transactions on Signal Processing, vol. 50, no. 2, pp.
        174-188, Feb. 2002.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # First loop: propagate characterizations and compute weights
        tmp_particles = []
        tmp_likelihoods = []
        for par in self.particles:

            # Compute characterization
            mu = self.propagate_sample(par[1], robot_forward_motion, robot_angular_motion)

            # Compute and store current particle's weight
            likelihood = self.compute_likelihood(mu, measurements, landmarks) 
            weight = likelihood * par[0]
            tmp_likelihoods.append(likelihood)

            # Store (notice mu will not be used later)
            tmp_particles.append([weight, mu])

        # Normalize particle weights
        tmp_particles = self.normalize_weights(tmp_particles)

        # Resample indices from propagated particles
        new_indices = self.sample_multinomial_indices(tmp_particles)

        # Second loop: now propagate the state of all particles indices that survived
        new_samples = []
        for idx in new_indices:

            # Get particle state associated with current index from original set of particles
            par = self.particles[idx]

            # Propagate the particle state
            propagated_state = self.propagate_sample(par[1], robot_forward_motion, robot_angular_motion)

            # Compute current particle's weight using the measurement likelihood of the characterization
            wi_tmp = tmp_likelihoods[idx]
            if wi_tmp < 1e-10:
                wi_tmp = 1e-10  # avoid division by zero
            weight = self.compute_likelihood(propagated_state, measurements, landmarks) / wi_tmp

            # Store
            new_samples.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_samples)
