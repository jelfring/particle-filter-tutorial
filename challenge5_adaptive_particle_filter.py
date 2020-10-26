#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR)
from core.resampling import ResamplingAlgorithms

# Load variables
from shared_simulation_settings import *

# Particle filters that will be compared
from core.particle_filters import AdaptiveParticleFilterKld, ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt

import numpy as np
import copy

if __name__ == '__main__':
    """
    This file demonstrates the difference between the SIR particle filter that has a constant number of particles and
    the adaptive particle filter that varies the number of particles on the fly. Afterwards, the number of particles
    for both particle filters are plotted over time together with the estimation error of the robot's x-position. The
    results show that the adaptive particle filter achieves a similar estimation accuracy with less particles once the
    particles have converged.
    """

    print("Starting adaptive particle filter demo.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(world_size_x, world_size_y, landmark_positions)
    visualizer = Visualizer()
    num_time_steps = 50

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
    # The process and measurement model noise is not equal to true noise.
    ##

    # Number of particles
    number_of_particles = 750

    # Initialize particle filter

    # Set resampling algorithm used (where applicable)
    resampling_algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Process model noise (values used in the paper)
    motion_model_forward_std = 0.20
    motion_model_turn_std = 0.05
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Initialize SIR particle filter
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise,
        resampling_algorithm)
    particle_filter_sir.initialize_particles_uniform()

    # Adaptive particle filter specific settings (see AdaptiveParticleFilterKld constructor for more extensive
    # documentation
    resolutions_grid = [0.2, 0.2, 0.3]
    epsilon = 0.15
    upper_quantile = 3
    min_number_of_particles = 50
    max_number_of_particles = 2e4

    # Initialize adaptive particle filter with the same set of particles
    adaptive_particle_filter_kld = AdaptiveParticleFilterKld(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise,
        resolutions_grid,
        epsilon,
        upper_quantile,
        min_number_of_particles,
        max_number_of_particles)
    adaptive_particle_filter_kld.set_particles(copy.deepcopy(particle_filter_sir.particles))

    ##
    # Start simulation
    ##
    errors_kld = []
    npar_kld = []
    errors_sir = []
    npar_sir = []
    for i in range(num_time_steps):


        # Simulate robot motion (required motion will not exactly be achieved)
        robot.move(desired_distance=robot_setpoint_motion_forward,
                   desired_rotation=robot_setpoint_motion_turn,
                   world=world)

        # Simulate measurement
        measurements = robot.measure(world)

        # Real robot pose
        robot_pose = np.array([robot.x, robot.y, robot.theta])

        # Update adaptive particle filter KLD
        adaptive_particle_filter_kld.update(robot_forward_motion=robot_setpoint_motion_forward,
                                            robot_angular_motion=robot_setpoint_motion_turn,
                                            measurements=measurements,
                                            landmarks=world.landmarks)
        errors_kld.append(robot_pose - np.asarray(adaptive_particle_filter_kld.get_average_state()))
        npar_kld.append(len(adaptive_particle_filter_kld.particles))

        # Update SIR particle filter
        particle_filter_sir.update(robot_forward_motion=robot_setpoint_motion_forward,
                                   robot_angular_motion=robot_setpoint_motion_turn,
                                   measurements=measurements,
                                   landmarks=world.landmarks)
        errors_sir.append(robot_pose - np.asarray(particle_filter_sir.get_average_state()))
        npar_sir.append(len(particle_filter_sir.particles))

    print(npar_kld)

    # Show results
    fontSize = 18
    plt.rcParams.update({'font.size': fontSize})
    plt.subplot(211)
    l1, = plt.plot([error[0] for error in errors_sir], 'k-')
    l2, = plt.plot([error[0] for error in errors_kld], 'r--')
    l1.set_label('Standard')
    l2.set_label('Adaptive')
    plt.legend()
    plt.ylabel('Error x-position [m]')
    plt.subplot(212)
    plt.plot([npar for npar in npar_sir], 'k-')
    plt.plot([npar for npar in npar_kld], 'r--')
    plt.yscale('symlog')
    plt.legend()
    plt.xlabel('Time index')
    plt.ylabel('Number of particles')
    plt.show()