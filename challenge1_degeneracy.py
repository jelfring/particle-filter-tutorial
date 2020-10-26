#!/usr/bin/env python

# Simulation + plotting requires a robot, visualizer and world
from simulator import Robot, Visualizer, World

# Supported resampling methods (resampling algorithm enum for SIR and SIR-derived particle filters)
from core.resampling import ResamplingAlgorithms

# Load variables
from shared_simulation_settings import *

# Particle filters
from core.particle_filters import ParticleFilterSIR

# For showing plots (plt.show())
import matplotlib.pyplot as plt


def never_resample():
    """
    Function that always returns false and thereby enables switching off resampling. This function is needed to switch
    off resampling.

    :return: Boolean that always equals false
    """
    return False


if __name__ == '__main__':

    """
    This file demonstrates the particle filter degeneracy problem that occurs in case a particle filter never resamples.
    """

    print("Starting demonstration of particle filter degeneracy.")

    ##
    # Set simulated world and visualization properties
    ##

    # Simulated world
    world = World(world_size_x, world_size_y, landmark_positions)

    # Initialize visualization
    show_particle_pose = False  # only set to true for low #particles (very slow)
    visualizer = Visualizer(show_particle_pose)

    # Number of simulated time steps
    n_time_steps = 15

    # Simulated robot
    robot = Robot(robot_initial_x_position,
                  robot_initial_y_position,
                  robot_initial_heading,
                  true_robot_motion_forward_std,
                  true_robot_motion_turn_std,
                  true_robot_meas_noise_distance_std,
                  true_robot_meas_noise_angle_std)

    # Number of particles
    number_of_particles = 500

    # Set resampling algorithm used
    resampling_algorithm = ResamplingAlgorithms.MULTINOMIAL

    # Initialize the particle filter

    # Initialize SIR particle filter
    particle_filter_sir = ParticleFilterSIR(
        number_of_particles,
        pf_state_limits,
        process_noise,
        measurement_noise,
        resampling_algorithm)
    particle_filter_sir.initialize_particles_uniform()

    # Turn OFF resampling
    particle_filter_sir.needs_resampling = never_resample

    # Start simulation
    max_weights = []
    for i in range(n_time_steps):

        # Simulate robot move, simulate measurement and update particle filter
        robot.move(robot_setpoint_motion_forward,
                   robot_setpoint_motion_turn,
                   world)

        measurements = robot.measure(world)
        particle_filter_sir.update(robot_setpoint_motion_forward,
                                   robot_setpoint_motion_turn,
                                   measurements,
                                   world.landmarks)

        # Show maximum normalized particle weight (converges to 1.0)
        w_max = particle_filter_sir.get_max_weight()
        max_weights.append(w_max)
        print("Time step {}: max weight: {}".format(i, w_max))

    # Plot weights as function of time step
    fontSize = 14
    plt.rcParams.update({'font.size': fontSize})
    plt.plot(range(n_time_steps), max_weights, 'k')
    plt.xlabel("Time index")
    plt.ylabel("Maximum particle weight")
    plt.xlim(0, n_time_steps-1)
    plt.ylim(0, 1.1)
    plt.show()
