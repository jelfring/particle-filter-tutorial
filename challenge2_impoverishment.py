#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt

if __name__ == '__main__':

    print("Starting demonstration of particle filter sample impoverishment.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(10.0, 10.0, [[2.0, 2.0], [2.0, 8.0], [9.0, 2.0], [8, 9]])
    visualizer = Visualizer()
    num_time_steps = 25

    ##
    # True robot properties (simulator settings)
    ##

    # Setpoint (desired) motion robot
    robot_setpoint_motion_forward = 0.25
    robot_setpoint_motion_turn = 0.02

    # True simulated robot motion is set point plus additive zero mean Gaussian noise
    true_robot_motion_forward_std = 0.005
    true_robot_motion_turn_std = 0.002

    # Robot measurements are corrupted by additive zero mean Gaussian measurement noise
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
    # The process and measurement model noise is not equal to true noise.
    ##

    number_of_particles = 500
    pf_state_limits = [0, world.x_max, 0, world.y_max]

    # Process model noise (zero mean additive Gaussian noise)
    # IMPOVERISHMENT: artificially set to unreasonably low value
    motion_model_forward_std = 0.003
    motion_model_turn_std = 0.003
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement model noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Initialize particle filter

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

    # Start simulation
    for i in range(num_time_steps):
        # Simulate robot move, simulate measurement and update particle filter
        robot.move(desired_distance=robot_setpoint_motion_forward,
                   desired_rotation=robot_setpoint_motion_turn,
                   world=world)
        measurements = robot.measure(world)
        particle_filter_sir.update(robot_forward_motion=robot_setpoint_motion_forward,
                                  robot_angular_motion=robot_setpoint_motion_turn,
                                  measurements=measurements,
                                  landmarks=world.landmarks)

        # Visualize weighted average all particles only
        visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    # Draw all particle at last time step
    visualizer.draw_world(world, robot, particle_filter_sir.particles, hold_on=True)

    plt.show()

