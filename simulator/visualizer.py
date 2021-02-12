# Plotting will be done with matplotlib
import matplotlib.pyplot as plt

# For generating plot more efficiently
import numpy as np

# Custom objects that must be visualized
from .world import World
from .robot import Robot


class Visualizer:
    """
    Class for visualing the world, the true robot pose and the discrete distribution that estimates the robot pose by
    means of a set of (weighted) particles.
    """

    def __init__(self, draw_particle_pose=False):
        """
        Initialize visualizer. By setting the flag to false the full 2D pose will be visualized. This makes
        visualization much slower hence is only recommended for a relatively low number of particles.

        :param draw_particle_pose: Flag that determines whether 2D positions (default) or poses must be visualized.
        """

        self.x_margin = 1
        self.y_margin = 1
        self.circle_radius_robot = 0.02  # 0.25
        self.draw_particle_pose = draw_particle_pose
        self.landmark_size = 6
        self.scale = 2  # meter / inch
        self.robot_arrow_length = 0.5 / self.scale

    def update_robot_radius(self, robot_radius):
        """
        Set the radius that must be used for drawing the robot in the simulated world.
        :param robot_radius:  new robot radius
        """
        self.circle_radius_robot = robot_radius
        self.robot_arrow_length = robot_radius

    def update_landmark_size(self, landmark_size):
        """
        Set the size that must be used for drawing the landmarks in the simulated world.
        :param landmark_size:  new landmark size
        """
        self.landmark_size = landmark_size

    def draw_world(self, world, robot, particles, hold_on=False, particle_color='g'):
        """
        Draw the simulated world with its landmarks, the robot 2D pose and the particles that represent the discrete
        probability distribution that estimates the robot pose.

        :param world: World object (includes dimensions and landmarks)
        :param robot: True robot 2D pose (x, y, heading)
        :param particles: Set of weighted particles (list of [weight, [x, y, heading]]-lists)
        :param hold_on: Boolean that indicates whether figure must be cleared or nog
        :param particle_color: Color used for particles (as matplotlib string)
        """

        # Dimensions world
        x_min = -self.x_margin
        x_max = self.x_margin + world.x_max
        y_min = -self.y_margin
        y_max = self.y_margin + world.y_max

        # Draw world
        plt.figure(1, figsize=((x_max-x_min) / self.scale, (y_max-y_min) / self.scale))
        if not hold_on:
            plt.clf()
        plt.plot([0, world.x_max], [0, 0], 'k-')                      # lower line
        plt.plot([0, 0], [0, world.y_max], 'k-')                      # left line
        plt.plot([0, world.x_max], [world.y_max, world.y_max], 'k-')  # top line
        plt.plot([world.x_max, world.x_max], [0, world.y_max], 'k-')  # right line

        # Set limits axes
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        # No ticks on axes
        plt.xticks([])
        plt.yticks([])

        # Title
        plt.title("{} particles".format(len(particles)))

        # Add landmarks
        landmarks = np.array(world.landmarks)
        plt.plot(landmarks[:, 0], landmarks[:, 1], 'bs', linewidth=2, markersize=self.landmark_size)

        # Add particles
        if self.draw_particle_pose:
            # Warning: this is very slow for large numbers of particles
            radius_scale_factor = len(particles) / 10.
            for p in particles:
                self.add_pose2d(p[1][0], p[1][1], p[1][2], 1, particle_color, radius_scale_factor * p[0])
        else:
            # Convert to numpy array for efficiency reasons (drop weights)
            states = np.array([np.array(state_i[1]) for state_i in particles])
            plt.plot(states[:, 0], states[:, 1], particle_color+'.', linewidth=1, markersize=2)

        # Add robot pose
        self.add_pose2d(robot.x, robot.y, robot.theta, 1, 'r', self.circle_radius_robot)

        plt.savefig('multimodal_one_landmark.png')

        # Show
        plt.pause(0.05)

    def add_pose2d(self, x, y, theta, fig_num, color, radius):
        """
        Plot a 2D pose in given figure with given color and radius (circle with line indicating heading).

        :param x: X-position (center circle).
        :param y: Y-position (center circle).
        :param theta: Heading (line from center circle with this heading will be added).
        :param fig_num: Figure in which pose must be added.
        :param color: Color of the lines.
        :param radius: Radius of the circle.
        :return:
        """

        # Select correct figure
        plt.figure(fig_num)

        # Draw circle at given position (higher 'zorder' value means draw later, hence on top of other lines)
        circle = plt.Circle((x, y), radius, facecolor=color, edgecolor=color, alpha=0.4, zorder=20)
        plt.gca().add_patch(circle)

        # Draw line indicating heading
        plt.plot([x, x + radius * np.cos(theta)],
                 [y, y + radius * np.sin(theta)], color)
