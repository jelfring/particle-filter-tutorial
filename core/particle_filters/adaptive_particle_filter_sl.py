from .particle_filter_base import ParticleFilter
from core.resampling import generate_sample_index


class AdaptiveParticleFilterSl(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 sum_likelihoods_threshold,
                 max_number_particles):
        """
        Initialize the adaptive particle filter using sum of likelihoods sampling proposed explained in [1,2].

        [1] Straka, Ondrej, and Miroslav Simandl. "A survey of sample size adaptation techniques for particle filters."
        IFAC Proceedings Volumes 42.10 (2009): 1358-1363.
        [2] Koller, Daphne, and Raya Fratkina. "Using Learning for Approximation in Stochastic Processes." ICML. 1998.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param sum_likelihoods_threshold: Minimum sum of all measurement likelihoods after update step.
        :param max_number_particles: Maximum number of particles that can be used.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set adaptive particle filter specific properties
        self.maximum_number_of_particles = max_number_particles
        self.sum_likelihoods_threshold = sum_likelihoods_threshold

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement. Continue core as long as the sum of all
        sampled particles is below a predefined threshold and the number of particles does not exceed the predefined
        maximum number of particles. The minimum number of particles is equal to the threhold since a measurement
        likelihoods can not exceed one by definition.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Store new samples
        new_particles = []

        sum_likelihoods = 0
        number_of_new_particles = 0
        while sum_likelihoods < self.sum_likelihoods_threshold and \
                number_of_new_particles < self.maximum_number_of_particles:

            # Get sample from discrete distribution given by particle weights
            index_j = generate_sample_index(self.particles)

            # Propagate state of selected particle
            propaged_state = self.propagate_sample(self.particles[index_j][1],
                                                   robot_forward_motion,
                                                   robot_angular_motion)

            # Compute the weight that this propagated state would get with the current measurement
            importance_weight = self.compute_likelihood(propaged_state, measurements, landmarks)

            # Add weighted particle to new particle set
            new_particles.append([importance_weight, propaged_state])

            # Administration
            sum_likelihoods += importance_weight
            number_of_new_particles += 1

        # Store new particle set and normalize weights
        self.particles = self.normalize_weights(new_particles)
