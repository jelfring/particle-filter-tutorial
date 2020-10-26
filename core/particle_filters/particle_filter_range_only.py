from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler

import numpy as np


class ParticleFilterRangeOnly(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations).
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm):
        """
        Initialize the SIR range measurement only particle filter. Largely copied from the SIR particle filter.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameter range (standard deviation): std_range.
        :param resampling_algorithm: Algorithm that must be used for core.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set SIR specific properties
        self.resampling_algorithm = resampling_algorithm
        self.resampler = Resampler()

    def needs_resampling(self):
        """
        Method that determines whether not a core step is needed for the current particle filter state estimate.
        The sampling importance core (SIR) scheme resamples every time step hence always return true.

        :return: Boolean indicating whether or not core is needed.
        """
        return True

    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute the importance weight p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark distance_to_landmark, unit is meters
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Importance weight
        """

        # Initialize measurement likelihood
        measurement_likelihood_sample = 1.0

        # Loop over all landmarks for current particle
        for i, lm in enumerate(landmarks):

            # Compute expected measurement assuming the current particle state
            dx = sample[0] - lm[0]
            dy = sample[1] - lm[1]
            expected_distance = np.sqrt(dx*dx + dy*dy)

            # Map difference true and expected distance measurement to probability
            p_z_given_x_distance = \
                np.exp(-(expected_distance-measurement[i]) * (expected_distance-measurement[i]) /
                       (2.0 * self.measurement_noise * self.measurement_noise))

            # Incorporate likelihoods current landmark
            measurement_likelihood_sample *= p_z_given_x_distance

        # Return importance weight based on all landmarks
        return measurement_likelihood_sample

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and resample if needed.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Loop over all particles
        new_particles = []
        for par in self.particles:

            # Propagate the particle state according to the current particle
            propagated_state = self.propagate_sample(par[1], robot_forward_motion, robot_angular_motion)

            # Compute current particle's weight
            weight = par[0] * self.compute_likelihood(propagated_state, measurements, landmarks)

            # Store
            new_particles.append([weight, propagated_state])

        # Update particles
        self.particles = self.normalize_weights(new_particles)

        # Resample if needed
        if self.needs_resampling():
            self.particles = self.resampler.resample(self.particles, self.n_particles, self.resampling_algorithm)
