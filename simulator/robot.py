import numpy as np

from .world import *


class Robot:

    def __init__(self, x, y, theta, std_forward, std_turn, std_meas_distance, std_meas_angle):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param x: Initial robot x-position (m)
        :param y: Initial robot y-position (m)
        :param theta: Initial robot heading (rad)
        :param std_forward: Standard deviation zero mean additive noise on forward motions (m)
        :param std_turn: Standard deviation zero mean Gaussian additive noise on turn actions (rad)
        :param std_meas_distance: Standard deviation zero mean Gaussian additive measurement noise (m)
        :param std_meas_angle: Standard deviation zero mean Gaussian additive measurement noise (rad)
        """

        # Initialize robot pose
        self.x = x
        self.y = y
        self.theta = theta

        # Set standard deviations noise robot motion
        self.std_forward = std_forward
        self.std_turn = std_turn

        # Set standard deviation measurements
        self.std_meas_distance = std_meas_distance
        self.std_meas_angle = std_meas_angle

    def move(self, desired_distance, desired_rotation, world):
        """
        Move the robot according to given arguments and within world of given dimensions. The true motion is the sum of
        the desired motion and additive Gaussian noise that represents the fact that the desired motion cannot exactly
        be realized, e.g., due to imperfect control and sensing.

        --- Suggestion for alternative implementation ------------------------------------------------------------------
        An alternative approach could be to replace the argument names of this function to true_distance_driven and
        true_rotation. These true motion could be applied to the robot state without adding noise:
          self.theta += true_rotation
          self.x += true_distance_driven * np.cos(self.theta)
          self.y += true_distance_driven * np.sin(self.theta)
        Then robot displacement measurements can be modelled as follows:
          measured_distance_driven = self._get_gaussian_noise_sample(desired_distance, self.std_forward)
          measured_angle_rotated = self._get_gaussian_noise_sample(desired_rotation, self.std_turn)
        And one can return noisy measurements:
          return [measured_distance_driven, measured_angle_rotated]
        that are then being used as input for the particle filter propagation step. This would obviously still require
        the cyclic world checks (on both measurement and true robot state). The particle filter estimation results will
        roughly be the same in terms of performance, however, this might be more intuitive for some of the readers.
        ----------------------------------------------------------------------------------------------------------------

        :param desired_distance: forward motion setpoint of the robot (m)
        :param desired_rotation: angular rotation setpoint of the robot (rad)
        :param world: dimensions of the cyclic world in which the robot executes its motion
        """

        # Compute relative motion (true motion is desired motion with some noise)
        distance_driven = self._get_gaussian_noise_sample(desired_distance, self.std_forward)
        angle_rotated = self._get_gaussian_noise_sample(desired_rotation, self.std_turn)

        # Update robot pose
        self.theta += angle_rotated
        self.x += distance_driven * np.cos(self.theta)
        self.y += distance_driven * np.sin(self.theta)

        # Cyclic world assumption (i.e. crossing right edge -> enter on left hand side)
        self.x = np.mod(self.x, world.x_max)
        self.y = np.mod(self.y, world.y_max)

        # Angles in [0, 2*pi]
        self.theta = np.mod(self.theta, 2*np.pi)

    def measure(self, world):
        """
        Perform a measurement. The robot is assumed to measure the distance to and angles with respect to all landmarks
        in meters and radians respectively. While doing so, the robot experiences zero mean additive Gaussian noise.

        :param world: World containing the landmark positions.
        :return: List of lists: [[dist_to_landmark1, angle_wrt_landmark1], dist_to_landmark2, angle_wrt_landmark2], ...]
        """

        # Loop over measurements
        measurements = []
        for lm in world.landmarks:
            dx = self.x - lm[0]
            dy = self.y - lm[1]

            # Measured distance perturbed by zero mean additive Gaussian noise
            z_distance = self._get_gaussian_noise_sample(np.sqrt(dx * dx + dy * dy), self.std_meas_distance)

            # Measured angle perturbed by zero mean additive Gaussian noise
            z_angle = self._get_gaussian_noise_sample(np.arctan2(dy, dx), self.std_meas_angle)

            # Store measurement
            measurements.append([z_distance, z_angle])

        return measurements

    @staticmethod
    def _get_gaussian_noise_sample(mu, sigma):
        """
        Get a random sample from a 1D Gaussian distribution with mean mu and standard deviation sigma.

        :param mu: mean of distribution
        :param sigma: standard deviation
        :return: random sample from distribution with given parameters
        """
        return np.random.normal(loc=mu, scale=sigma, size=1)[0]
