import numpy as np

from .robot import *
from .world import *


class RobotRange(Robot):

    def __init__(self, x, y, theta, std_forward, std_turn, std_meas_distance):
        """
        Initialize the robot with given 2D pose. In addition set motion uncertainty parameters.

        :param x: Initial robot x-position (m)
        :param y: Initial robot y-position (m)
        :param theta: Initial robot heading (rad)
        :param std_forward: Standard deviation zero mean additive noise on forward motions (m)
        :param std_turn: Standard deviation zero mean Gaussian additive noise on turn actions (rad)
        :param std_meas_distance: Standard deviation zero mean Gaussian additive measurement noise (m)
        """

        # Initialize robot base class without standard deviation for angular measurements
        Robot.__init__(self, x, y, theta, std_forward, std_turn, std_meas_distance, None)

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

            # Store measurement
            measurements.append(z_distance)

        return measurements
