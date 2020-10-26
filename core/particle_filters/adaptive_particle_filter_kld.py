import numpy as np

from .particle_filter_base import ParticleFilter

from core.resampling import generate_sample_index, compute_required_number_of_particles_kld


class AdaptiveParticleFilterKld(ParticleFilter):
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
                 resolutions,
                 epsilon,
                 upper_quantile,
                 min_number_particles,
                 max_number_particles):
        """
        Initialize the adaptive particle filter using Kullback-Leibler divergence (KLD) sampling proposed in [1].

        [1] Fox, Dieter. "Adapting the sample size in particle filters through KLD-sampling." The international Journal
        of robotics research 22.12 (2003): 985-1003.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resolutions: Resolution of 3D-histogram used to approximate the true posterior distribution
        (x, y, theta).
        :param epsilon: Maximum allowed distance (error) between true and estimated distribution when computing number
        of required particles.
        :param upper_quantile: Upper standard normal distribution quantile for (1-delta) where delta is the probability.
        that the error on the estimated distribution will be less than epsilon.
        :param min_number_particles: Minimum number of particles that must be used.
        :param max_number_particles: Maximum number of particles that can be used.
        """
        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Set adaptive particle filter specific properties
        self.resolutions = resolutions
        self.epsilon = epsilon
        self.upper_quantile = upper_quantile
        self.minimum_number_of_particles = int(min_number_particles)
        self.maximum_number_of_particles = int(max_number_particles)

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and adopting the Kullback-Leibler divergence
        sampling method (KLD-sampling) proposed by Dieter Fox.

        Assumption in this implementation: world starts at x=0, y=0, theta=0 (for mapping particle state to histogram
        bin).

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Store new samples
        new_particles = []

        bins_with_support = []
        number_of_new_particles = 0
        number_of_bins_with_support = 0
        number_of_required_particles = self.minimum_number_of_particles
        while number_of_new_particles < number_of_required_particles:

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
            number_of_new_particles += 1

            # Next, we convert the discrete distribution of all new samples into a histogram. We must check if the new
            # state (propagated_state) falls in a histogram bin with support or in an empty bin. We keep track of the
            # number of bins with support. Instead of adopting a (more efficient) tree, a simple list is used to
            # store all bin indices with support since there is are no performance requirements for our use case.

            # Map state to bin indices
            indices = [np.floor(propaged_state[0] / self.resolutions[0]),
                       np.floor(propaged_state[1] / self.resolutions[1]),
                       np.floor(propaged_state[2] / self.resolutions[2])]

            # Add indices if this bin is empty (i.e. is not in list yet)
            if indices not in bins_with_support:
                bins_with_support.append(indices)
                number_of_bins_with_support += 1

            # Update number of required particles (only defined if number of bins with support above 1)
            if number_of_bins_with_support > 1:
                number_of_required_particles = compute_required_number_of_particles_kld(number_of_bins_with_support,
                                                                                        self.epsilon,
                                                                                        self.upper_quantile)

            # Make sure number of particles constraints are not violated
            number_of_required_particles = max(number_of_required_particles, self.minimum_number_of_particles)
            number_of_required_particles = min(number_of_required_particles, self.maximum_number_of_particles)

        # Store new particle set and normalize weights
        self.particles = self.normalize_weights(new_particles)
