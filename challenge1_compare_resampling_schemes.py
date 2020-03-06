#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterSIR, ParticleFilterNEPR, ParticleFilterMWR

# For computing errors
import numpy as np

if __name__ == '__main__':

    print("Compare three resampling schemes.")

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
    errors_sir = []
    errors_nepr = []
    errors_mwr = []
    cnt_sir = 0
    cnt_nepr = 0
    cnt_mwr = 0
    for trial in range(n_trials):
        print("Trial: ", trial)
        # Initialize SIR particle filter: resample every time step
        particle_filter_sir = ParticleFilterSIR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=algorithm)
        particle_filter_sir.initialize_particles_uniform()

        # Resample if approximate number effective particle drops below n_particles / 4
        particle_filter_nepr = ParticleFilterNEPR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=algorithm,
            number_of_effective_particles_threshold=number_of_particles / 4.0)
        particle_filter_nepr.set_particles(particle_filter_sir.particles)

        # Resample based on reciprocal of maximum particle weight drops below 1 / 0.005
        particle_filter_mwr = ParticleFilterMWR(
            number_of_particles=number_of_particles,
            limits=pf_state_limits,
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            resampling_algorithm=algorithm,
            resampling_threshold=1.0 / 0.005)
        particle_filter_mwr.set_particles(particle_filter_sir.particles)

        for i in range(n_time_steps):

            # Simulate robot motion (required motion will not exactly be achieved)
            robot.move(desired_distance=robot_setpoint_motion_forward,
                       desired_rotation=robot_setpoint_motion_turn,
                       world=world)

            # Simulate measurement
            measurements = robot.measure(world)

            # Update SIR particle filter (in this case: propagate + weight update + resample)
            res = particle_filter_sir.update(robot_forward_motion=robot_setpoint_motion_forward,
                                       robot_angular_motion=robot_setpoint_motion_turn,
                                       measurements=measurements,
                                       landmarks=world.landmarks)
            if res:
                cnt_sir += 1

            # Update NEPR particle filter (in this case: propagate + weight update, resample if needed)
            res = particle_filter_nepr.update(robot_forward_motion=robot_setpoint_motion_forward,
                                        robot_angular_motion=robot_setpoint_motion_turn,
                                        measurements=measurements,
                                        landmarks=world.landmarks)
            if res:
                cnt_nepr += 1

            # Update MWR particle filter (in this case: propagate + weight update, resample if needed)
            res = particle_filter_mwr.update(robot_forward_motion=robot_setpoint_motion_forward,
                                       robot_angular_motion=robot_setpoint_motion_turn,
                                       measurements=measurements,
                                       landmarks=world.landmarks)
            if res:
                cnt_mwr += 1

            # Check if outputs are similar
            robot_pose = np.array([robot.x, robot.y, robot.theta])
            e_sir = robot_pose - np.asarray(particle_filter_sir.get_average_state())
            e_nepr = robot_pose - np.asarray(particle_filter_nepr.get_average_state())
            e_mwr = robot_pose - np.asarray(particle_filter_mwr.get_average_state())

            errors_sir.append(np.linalg.norm(e_sir))
            errors_nepr.append(np.linalg.norm(e_nepr))
            errors_mwr.append(np.linalg.norm(e_mwr))

    print("SIR mean error: {}, std error: {}".format(np.mean(np.asarray(errors_sir)), np.std(np.asarray(errors_sir))))
    print("NEPR mean error: {}, std error: {}".format(np.mean(np.asarray(errors_nepr)), np.std(np.asarray(errors_nepr))))
    print("MWR mean error: {}, std error: {}".format(np.mean(np.asarray(errors_mwr)), np.std(np.asarray(errors_mwr))))

    print("#updates in {} trials: {}, {}, {}".format(n_trials, cnt_sir, cnt_nepr, cnt_mwr))
