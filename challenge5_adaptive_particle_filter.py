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
    show_particle_pose = False  # only set to true for low #particles (very slow)
    visualizer = Visualizer(show_particle_pose)

    # Number of simulated time steps
    n_time_steps = 50

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

    ##
    # Particle filter settings
    ##

    number_of_particles = 750
    pf_state_limits = [0, world.x_max, 0, world.y_max]

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_forward_std = 0.20
    motion_model_turn_std = 0.05
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Set resampling algorithm used (where applicable)
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize SIR particle filter
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles=number_of_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resampling_algorithm=algorithm)
    particle_filter_sir.initialize_particles_uniform()

    # Initialize adaptive particle filter (KLD sampling)
    adaptive_particle_filter_kld = AdaptiveParticleFilterKld(
        number_of_particles=number_of_particles,
        limits=pf_state_limits,
        process_noise=process_noise,
        measurement_noise=measurement_noise,
        resolutions=[0.2, 0.2, 0.3],
        epsilon=0.15,
        upper_quantile=3,
        min_number_particles=50,
        max_number_particles=2e4)
    adaptive_particle_filter_kld.set_particles(copy.deepcopy(particle_filter_sir.particles))

    ##
    # Start simulation
    ##
    errors_kld = []
    npar_kld = []
    errors_sir = []
    npar_sir = []
    for i in range(n_time_steps):


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