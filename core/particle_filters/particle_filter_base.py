from abc import abstractmethod
import copy
import numpy as np


class ParticleFilter:
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Abstract class
    """

    def __init__(self, number_of_particles, limits, process_noise, measurement_noise):
        """
        Initialize the abstract particle filter.

        :param number_of_particles: Number of particles
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax]
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular]
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle]
        """

        if number_of_particles < 1:
            print("Warning: initializing particle filter with number of particles < 1: {}".format(number_of_particles))

        # Initialize filter settings
        self.n_particles = number_of_particles
        self.particles = []

        # State related settings
        self.state_dimension = 3  # x, y, theta
        self.x_min = limits[0]
        self.x_max = limits[1]
        self.y_min = limits[0]
        self.y_max = limits[1]

        # Set noise
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        self.particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):
            # Add particle i
            self.particles.append(
                [weight, [
                    np.random.uniform(self.x_min, self.x_max, 1)[0],
                    np.random.uniform(self.y_min, self.y_max, 1)[0],
                    np.random.uniform(0, 2 * np.pi, 1)[0]]
                 ]
            )

    def initialize_particles_gaussian(self, mean_vector, standard_deviation_vector):
        """
        Initialize particle filter using a Gaussian distribution with dimension three: x, y, heading. Only standard
        deviations can be provided hence the covariances are all assumed zero.

        :param mean_vector: Mean of the Gaussian distribution used for initializing the particle states
        :param standard_deviation_vector: Standard deviations (one for each dimension)
        :return: Boolean indicating success
        """

        # Check input dimensions
        if len(mean_vector) != self.state_dimension or len(standard_deviation_vector) != self.state_dimension:
            print("Means and state deviation vectors have incorrect length in initialize_particles_gaussian()")
            return False

        # Initialize particles with uniform weight distribution
        self.particles = []
        weight = 1.0 / self.n_particles
        for i in range(self.n_particles):

            # Get state sample
            state_i = np.random.normal(mean_vector, standard_deviation_vector, self.state_dimension).tolist()

            # Add particle i
            self.particles.append([weight, self.validate_state(state_i)])

    def validate_state(self, state):
        """
        Validate the state. State values outide allowed ranges will be corrected for assuming a 'cyclic world'.

        :param state: Input particle state.
        :return: Validated particle state.
        """

        # Make sure state does not exceed allowed limits (cyclic world)
        while state[0] < self.x_min:
            state[0] += (self.x_max - self.x_min)
        while state[0] > self.x_max:
            state[0] -= (self.x_max - self.x_min)
        while state[1] < self.y_min:
            state[1] += (self.y_max - self.y_min)
        while state[1] > self.y_max:
            state[1] -= (self.y_max - self.y_min)

        # Angle must be [-pi, pi]
        while state[2] > np.pi:
            state[2] -= 2 * np.pi
        while state[2] < -np.pi:
            state[2] += 2 * np.pi

        return state

    def set_particles(self, particles):
        """
        Initialize the particle filter using the given set of particles.

        :param particles: Initial particle set: [[weight_1, [x1, y1, theta1]], ..., [weight_n, [xn, yn, thetan]]]
        """

        # Assumption: particle have correct format, set particles
        self.particles = copy.deepcopy(particles)
        self.n_particles = len(self.particles)

    def get_average_state(self):
        """
        Compute average state according to all weighted particles

        :return: Average x-position, y-position and orientation
        """

        # Compute sum of all weights
        sum_weights = 0.0
        for weighted_sample in self.particles:
            sum_weights += weighted_sample[0]

        # Compute weighted average
        avg_x = 0.0
        avg_y = 0.0
        avg_theta = 0.0
        for weighted_sample in self.particles:
            avg_x += weighted_sample[0] / sum_weights * weighted_sample[1][0]
            avg_y += weighted_sample[0] / sum_weights * weighted_sample[1][1]
            avg_theta += weighted_sample[0] / sum_weights * weighted_sample[1][2]

        return [avg_x, avg_y, avg_theta]

    def get_max_weight(self):
        """
        Find maximum weight in particle filter.

        :return: Maximum particle weight
        """
        return max([weighted_sample[0] for weighted_sample in self.particles])


    def print_particles(self):
        """
        Print all particles: index, state and weight.
        """

        print("Particles:")
        for i in range(self.n_particles):
            print(" ({}): {} with w: {}".format(i+1, self.particles[i][1], self.particles[i][0]))

    @staticmethod
    def normalize_weights(weighted_samples):
        """
        Normalize all particle weights.
        """

        # Compute sum weighted samples
        sum_weights = 0.0
        for weighted_sample in weighted_samples:
            sum_weights += weighted_sample[0]

        # Check if weights are non-zero
        if sum_weights < 1e-15:
            print("Weight normalization failed: sum of all weights is {} (weights will be reinitialized)".format(sum_weights))

            # Set uniform weights
            return [[1.0 / len(weighted_samples), weighted_sample[1]] for weighted_sample in weighted_samples]

        # Return normalized weights
        return [[weighted_sample[0] / sum_weights, weighted_sample[1]] for weighted_sample in weighted_samples]

    def propagate_sample(self, sample, forward_motion, angular_motion):
        """
        Propagate an individual sample with a simple motion model that assumes the robot rotates angular_motion rad and
        then moves forward_motion meters in the direction of its heading. Return the propagated sample (leave input
        unchanged).

        :param sample: Sample (unweighted particle) that must be propagated
        :param forward_motion: Forward motion in meters
        :param angular_motion: Angular motion in radians
        :return: propagated sample
        """
        # 1. rotate by given amount plus additive noise sample (index 1 is angular noise standard deviation)
        propagated_sample = copy.deepcopy(sample)
        propagated_sample[2] += np.random.normal(angular_motion, self.process_noise[1], 1)[0]

        # Compute forward motion by combining deterministic forward motion with additive zero mean Gaussian noise
        forward_displacement = np.random.normal(forward_motion, self.process_noise[0], 1)[0]

        # 2. move forward
        propagated_sample[0] += forward_displacement * np.cos(propagated_sample[2])
        propagated_sample[1] += forward_displacement * np.sin(propagated_sample[2])

        # Make sure we stay within cyclic world
        return self.validate_state(propagated_sample)

    def compute_likelihood(self, sample, measurement, landmarks):
        """
        Compute likelihood p(z|sample) for a specific measurement given sample state and landmarks.

        :param sample: Sample (unweighted particle) that must be propagated
        :param measurement: List with measurements, for each landmark [distance_to_landmark, angle_wrt_landmark], units
        are meters and radians
        :param landmarks: Positions (absolute) landmarks (in meters)
        :return Likelihood
        """

        # Initialize measurement likelihood
        likelihood_sample = 1.0

        # Loop over all landmarks for current particle
        for i, lm in enumerate(landmarks):

            # Compute expected measurement assuming the current particle state
            dx = sample[0] - lm[0]
            dy = sample[1] - lm[1]
            expected_distance = np.sqrt(dx*dx + dy*dy)
            expected_angle = np.arctan2(dy, dx)

            # Map difference true and expected distance measurement to probability
            p_z_given_x_distance = \
                np.exp(-(expected_distance-measurement[i][0]) * (expected_distance-measurement[i][0]) /
                       (2 * self.measurement_noise[0] * self.measurement_noise[0]))

            # Map difference true and expected angle measurement to probability
            p_z_given_x_angle = \
                np.exp(-(expected_angle-measurement[i][1]) * (expected_angle-measurement[i][1]) /
                       (2 * self.measurement_noise[1] * self.measurement_noise[1]))

            # Incorporate likelihoods current landmark
            likelihood_sample *= p_z_given_x_distance * p_z_given_x_angle

        # Return importance weight based on all landmarks
        return likelihood_sample

    @abstractmethod
    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement. Abstract method that must be implemented in derived
        class.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        pass
