#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Load variables
from shared_simulation_settings import *

# Particle filter
from core.particle_filters import KalmanParticleFilter

# For showing plots (plt.show())
import matplotlib.pyplot as plt

if __name__ == '__main__':

    """
    This file demonstrates how the extended Kalman particle filter works. No divergence occurs even though the number of 
    particles is low. In fact, the number of particles equals the number of particles in the divergence example.
    
    Note that the only the mean value for each particle is visualized (not the covariance associated with the mean 
    value).
    """

    print("Starting demonstration of extended Kalman particle filter.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(world_size_x, world_size_y, landmark_positions)
    visualizer = Visualizer()
    num_time_steps = 30

    # Initialize simulated robot
    robot = Robot(x=world.x_max * 0.75,
                  y=world.y_max / 5.0,
                  theta=3.14 / 2.0,
                  std_forward=true_robot_motion_forward_std,
                  std_turn=true_robot_motion_turn_std,
                  std_meas_distance=true_robot_meas_noise_distance_std,
                  std_meas_angle=true_robot_meas_noise_angle_std)

    ##
    # Particle filter settings
    ##

    # Demonstrate divergence -> set number of particles too low
    number_of_particles = 100  # instead of 500 or even 1000

    # Initialize particle filter

    # Initialize extended Kalman particle filter
    kalman_particle_filter = KalmanParticleFilter(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise)
    kalman_particle_filter.initialize_particles_uniform()

    ##
    # Start simulation
    ##
    for i in range(num_time_steps):
        # Simulate robot move, perform measurements and update extended Kalman particle filter
        robot.move(robot_setpoint_motion_forward,
                   robot_setpoint_motion_turn,
                   world)
        measurements = robot.measure(world)
        kalman_particle_filter.update(robot_setpoint_motion_forward,
                                      robot_setpoint_motion_turn,
                                      measurements,
                                      world.landmarks)

        # Visualize particles after initialization (to avoid cluttered visualization)
        visualizer.draw_world(world, robot, kalman_particle_filter.particles, hold_on=True)

    # Draw all particle at last time step
    visualizer.draw_world(world, robot, kalman_particle_filter.particles, hold_on=True)

    plt.show()