#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods
from core.resampling import ResamplingAlgorithms

# Load variables
from shared_simulation_settings import *

# Particle filter
from core.particle_filters import ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """
    This particle filter demonstrates particle filter sample impoverishment. In order to enforce imporerishment, the
    process model noise is artificially lowered.
    """

    print("Starting demonstration of particle filter sample impoverishment.")

    ##
    # Set simulated world and visualization properties
    ##

    # Simulated world
    world = World(world_size_x, world_size_y, landmark_positions)

    # Visualizer
    visualizer = Visualizer()

    # Number of simulated time steps
    num_time_steps = 25

    # Initialize simulated robot
    robot = Robot(x=world.x_max * 0.75,
                  y=world.y_max / 5.0,
                  theta=3.14 / 2.0,
                  std_forward=true_robot_motion_forward_std,
                  std_turn=true_robot_motion_turn_std,
                  std_meas_distance=true_robot_meas_noise_distance_std,
                  std_meas_angle=true_robot_meas_noise_angle_std)

    # Number of particles
    number_of_particles = 500

    # IMPOVERISHMENT: artificially set to unreasonably low value for process model noise
    # Note we are redefining the common settings used in our shared simulation settings file.
    motion_model_forward_std = 0.003
    motion_model_turn_std = 0.003
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Resampling algorithm used
    resampling_algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise,
        resampling_algorithm)
    particle_filter_sir.initialize_particles_uniform()

    # Start simulation
    for i in range(num_time_steps):

        # Simulate robot move, simulate measurement and update particle filter
        robot.move(robot_setpoint_motion_forward,
                   robot_setpoint_motion_turn,
                   world)
        measurements = robot.measure(world)
        particle_filter_sir.update(robot_setpoint_motion_forward,
                                   robot_setpoint_motion_turn,
                                   measurements,world.landmarks)

        # Visualize weighted average all particles only
        visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    # Draw all particle at last time step
    visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    plt.show()

