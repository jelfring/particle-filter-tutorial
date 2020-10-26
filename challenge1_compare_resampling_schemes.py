#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods
from core.resampling import ResamplingAlgorithms

# Particle filters
from core.particle_filters import ParticleFilterSIR, ParticleFilterNEPR, ParticleFilterMWR

# Load variables
from shared_simulation_settings import *

# For computing errors
import numpy as np

if __name__ == '__main__':

    """
    In this program three particle filter will be used for exactly the same problem. The filters are identical except
    for the resampling strategy.
    1) The first particle filter resamples at every time step (SIR)
    2) The second particle filter resamples if the approximate number of effective particles drops below a pre-defined
       threshold (NEPR)
    3) The third particle filter resamples in case the reciprocal of the maximum weight drops below a pre-defined
       threshold (MWR)
    """

    print("Compare three resampling schemes.")

    # Define simulated world
    world = World(world_size_x, world_size_y, landmark_positions)

    # Initialize simulated robot
    robot = Robot(x=world.x_max * 0.75,
                  y=world.y_max / 5.0,
                  theta=3.14 / 2.0,
                  std_forward=true_robot_motion_forward_std,
                  std_turn=true_robot_motion_turn_std,
                  std_meas_distance=true_robot_meas_noise_distance_std,
                  std_meas_angle=true_robot_meas_noise_angle_std)

    # Set resampling algorithm used (same for all filters)
    resampling_algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Number of particles in the particle filters
    number_of_particles = 1000

    # Resampling thresholds
    number_of_effective_particles_threshold = number_of_particles / 4.0
    reciprocal_max_weight_resampling_threshold = 1.0 / 0.005

    ##
    # Simulation settings
    ##
    n_time_steps = 50  # Number of simulated time steps
    n_trials = 100     # Number of times each simulation will be repeated

    # Bookkeeping variables
    errors_sir = []
    errors_nepr = []
    errors_mwr = []
    cnt_sir = 0
    cnt_nepr = 0
    cnt_mwr = 0

    # Start main simulation loop
    for trial in range(n_trials):
        print("Trial: ", trial)

        # (Re)initialize SIR particle filter: resample every time step
        particle_filter_sir = ParticleFilterSIR(
            number_of_particles,
            pf_state_limits,
            process_noise,
            measurement_noise,
            resampling_algorithm)
        particle_filter_sir.initialize_particles_uniform()

        # Resample if approximate number effective particle drops below threshold
        particle_filter_nepr = ParticleFilterNEPR(
            number_of_particles,
            pf_state_limits,
            process_noise,
            measurement_noise,
            resampling_algorithm,
            number_of_effective_particles_threshold)
        particle_filter_nepr.set_particles(particle_filter_sir.particles)

        # Resample based on reciprocal of maximum particle weight drops below threshold
        particle_filter_mwr = ParticleFilterMWR(
            number_of_particles,
            pf_state_limits,
            process_noise,
            measurement_noise,
            resampling_algorithm,
            reciprocal_max_weight_resampling_threshold)
        particle_filter_mwr.set_particles(particle_filter_sir.particles)

        for i in range(n_time_steps):

            # Move the simulated robot
            robot.move(robot_setpoint_motion_forward,
                       robot_setpoint_motion_turn,
                       world)

            # Simulate measurement
            measurements = robot.measure(world)

            # Update SIR particle filter (in this case: propagate + weight update + resample)
            res = particle_filter_sir.update(robot_setpoint_motion_forward,
                                             robot_setpoint_motion_turn,
                                             measurements,
                                             world.landmarks)
            if res:
                cnt_sir += 1

            # Update NEPR particle filter (in this case: propagate + weight update, resample if needed)
            res = particle_filter_nepr.update(robot_setpoint_motion_forward,
                                              robot_setpoint_motion_turn,
                                              measurements,
                                              world.landmarks)
            if res:
                cnt_nepr += 1

            # Update MWR particle filter (in this case: propagate + weight update, resample if needed)
            res = particle_filter_mwr.update(robot_setpoint_motion_forward,
                                             robot_setpoint_motion_turn,
                                             measurements,
                                             world.landmarks)
            if res:
                cnt_mwr += 1

            # Compute errors
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
