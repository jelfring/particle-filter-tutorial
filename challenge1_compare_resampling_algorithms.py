#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterSIR

# For computing errors
import numpy as np

if __name__ == '__main__':

    print("Compare four resampling algorithms.")

    ##
    # Set simulated world and visualization properties
    ##
    world = World(10.0, 10.0, [[2.0, 2.0], [2.0, 8.0], [9.0, 2.0], [8, 9]])

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

    number_of_particles = 1000
    pf_state_limits = [0, world.x_max, 0, world.y_max]

    # Process model noise (zero mean additive Gaussian noise)
    motion_model_forward_std = 0.10
    motion_model_turn_std = 0.02
    process_noise = [motion_model_forward_std, motion_model_turn_std]

    # Measurement noise (zero mean additive Gaussian noise)
    meas_model_distance_std = 0.4
    meas_model_angle_std = 0.3
    measurement_noise = [meas_model_distance_std, meas_model_angle_std]

    # Set resampling algorithm used
    algorithm = ResamplingAlgorithms.MULTINOMIAL

    ##
    # Start simulation
    ##
    n_trials = 100
    errors_mult = []
    errors_sys = []
    errors_str = []
    errors_res = []
    for trial in range(n_trials):
        print("Trial: ", trial)
        # Multinomial resampling
        particle_filter_multinomial = ParticleFilterSIR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=ResamplingAlgorithms.MULTINOMIAL)
        particle_filter_multinomial.initialize_particles_uniform()

        # Systematic resampling
        particle_filter_systematic = ParticleFilterSIR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=ResamplingAlgorithms.SYSTEMATIC)
        particle_filter_systematic.set_particles(particle_filter_multinomial.particles)

        # Stratified resampling
        particle_filter_stratified = ParticleFilterSIR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=ResamplingAlgorithms.STRATIFIED)
        particle_filter_stratified.set_particles(particle_filter_multinomial.particles)

        # Residual resampling
        particle_filter_residual = ParticleFilterSIR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=ResamplingAlgorithms.RESIDUAL)
        particle_filter_residual.set_particles(particle_filter_multinomial.particles)

        for i in range(n_time_steps):

            # Simulate robot motion (required motion will not exactly be achieved)
            robot.move(desired_distance=robot_setpoint_motion_forward,
                       desired_rotation=robot_setpoint_motion_turn,
                       world=world)

            # Simulate measurement
            measurements = robot.measure(world)

            # Multinomial resampling PF
            particle_filter_multinomial.update(robot_forward_motion=robot_setpoint_motion_forward,
                                               robot_angular_motion=robot_setpoint_motion_turn,
                                               measurements=measurements,
                                               landmarks=world.landmarks)

            # Systematic resampling PF
            particle_filter_systematic.update(robot_forward_motion=robot_setpoint_motion_forward,
                                              robot_angular_motion=robot_setpoint_motion_turn,
                                              measurements=measurements,
                                              landmarks=world.landmarks)

            # Stratified resampling PF
            particle_filter_stratified.update(robot_forward_motion=robot_setpoint_motion_forward,
                                              robot_angular_motion=robot_setpoint_motion_turn,
                                              measurements=measurements,
                                              landmarks=world.landmarks)

            # Residual resampling PF
            particle_filter_residual.update(robot_forward_motion=robot_setpoint_motion_forward,
                                            robot_angular_motion=robot_setpoint_motion_turn,
                                            measurements=measurements,
                                            landmarks=world.landmarks)


            # Compute errors
            robot_pose = np.array([robot.x, robot.y, robot.theta])
            e_mult = robot_pose - np.asarray(particle_filter_multinomial.get_average_state())
            e_sys = robot_pose - np.asarray(particle_filter_systematic.get_average_state())
            e_str = robot_pose - np.asarray(particle_filter_stratified.get_average_state())
            e_res = robot_pose - np.asarray(particle_filter_residual.get_average_state())

            errors_mult.append(np.linalg.norm(e_mult))
            errors_sys.append(np.linalg.norm(e_sys))
            errors_str.append(np.linalg.norm(e_str))
            errors_res.append(np.linalg.norm(e_res))

    print("Multinomial mean error: {}, std error: {}".format(np.mean(np.asarray(errors_mult)), np.std(np.asarray(errors_mult))))
    print("Systematic mean error: {}, std error: {}".format(np.mean(np.asarray(errors_sys)), np.std(np.asarray(errors_sys))))
    print("Stratified mean error: {}, std error: {}".format(np.mean(np.asarray(errors_str)), np.std(np.asarray(errors_str))))
    print("Residual mean error: {}, std error: {}".format(np.mean(np.asarray(errors_res)), np.std(np.asarray(errors_res))))