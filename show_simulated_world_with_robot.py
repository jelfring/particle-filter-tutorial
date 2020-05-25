#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import AdaptiveParticleFilterKld, ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt

import numpy as np
import copy

if __name__ == '__main__':

    print("Starting adaptive particle filter demo.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(10.0, 10.0, [[2.0, 2.0], [2.0, 8.0], [9.0, 2.0], [8, 9]])

    # Initialize visualization
    show_particle_pose = True  # only set to true for low #particles (very slow)
    visualizer = Visualizer(show_particle_pose)

    ##
    # True robot properties (simulator settings)
    ##

    # Setpoint (desired) motion robot
    robot_setpoint_motion_forward = 0.25
    robot_setpoint_motion_turn = 0.02

    # True simulated robot motion is set point plus additive zero mean Gaussian noise with these standard deviation
    true_robot_motion_forward_std = 0.005
    true_robot_motion_turn_std = 0.002

    # Robot measurements are corrupted by measurement noise
    true_robot_meas_noise_distance_std = 0.2
    true_robot_meas_noise_angle_std = 0.05

    # Initialize simulated robot
    robot = Robot(x=world.x_max * 0.75,
                  y=world.y_max / 5.0,
                  theta=3.14 / 2.0,
                  std_forward=true_robot_motion_forward_std,
                  std_turn=true_robot_motion_turn_std,
                  std_meas_distance=true_robot_meas_noise_distance_std,
                  std_meas_angle=true_robot_meas_noise_angle_std)

    # Show world
    visualizer.update_robot_radius(0.4)
    visualizer.update_landmark_size(9)
    visualizer.draw_world(world, robot, [], hold_on=True, particle_color='g')
    plt.show()