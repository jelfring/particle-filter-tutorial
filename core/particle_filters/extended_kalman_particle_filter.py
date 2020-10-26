from .particle_filter_base import ParticleFilter
from core.resampling.resampler import Resampler

import copy
import numpy as np
from scipy.stats import multivariate_normal
from scipy import linalg


class KalmanParticleFilter(ParticleFilter):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * propagation and measurement models are largely hardcoded (except for standard deviations).

        For convenience reasons this class inherits from the generic particle filter class. Note however, that some of
        the methods are hardcoded and some of the members are unused.
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise):
        """
        Initialize the extended Kalman particle filter. Resampling method is hardcoded hence no argument.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        """

        # Initialize particle filter base class
        ParticleFilter.__init__(self, number_of_particles, limits, process_noise, measurement_noise)

        # Initialize covariance matrices for the EKF
        self.Q = np.diag([process_noise[0], process_noise[0], process_noise[1]])
        self.R = np.diag([measurement_noise[0], measurement_noise[1]])

    def initialize_particles_uniform(self):
        """
        Initialize the particles uniformly over the world assuming a 3D state (x, y, heading). No arguments are required
        and function always succeeds hence no return value.
        """

        # Initialize particles with uniform weight distribution
        super(KalmanParticleFilter, self).initialize_particles_uniform()

        # Add covariance matrices to particles (now particle is following list: [weight, particle_state,
        # covariance_matrix])
        for particle in self.particles:
            particle.append(np.eye(3))

    def multinomial_resampling(self):
        """
        Particles are sampled with replacement proportional to their weight and in arbitrary order. This leads
        to a maximum variance on the number of times a particle will be resampled, since any particle will be
        resampled between 0 and N times.

        This function is reimplemented in this class since each particle now contains three elements instead of two
        (covariance of extended Kalman filter is added).
        """

        # Get list with only weights
        weights = [weighted_sample[0] for weighted_sample in self.particles]

        # Compute cumulative sum
        Q = np.cumsum(weights).tolist()

        # As long as the number of new particles is insufficient
        n = 0
        new_particles = []
        while n < self.n_particles:

            # Draw a random sample u
            u = np.random.uniform(1e-6, 1, 1)[0]

            # Naive search
            m = 0
            while Q[m] < u:
                m += 1

            # Add copy of the particle  but set uniform weight
            new_sample = copy.deepcopy(self.particles[m])
            new_sample[0] = 1.0/self.n_particles
            new_particles.append(new_sample)

            # Added another sample
            n += 1

        self.particles = new_particles

    def update(self, robot_forward_motion, robot_angular_motion, measurements, landmarks):
        """
        Process a measurement given the measured robot displacement and resample.

        :param robot_forward_motion: Measured forward robot motion in meters.
        :param robot_angular_motion: Measured angular robot motion in radians.
        :param measurements: Measurements.
        :param landmarks: Landmark positions.
        """

        # Loop over all particles
        new_particles = []
        sum_weights = 0.0
        for par in self.particles:

            # This time an extended Kalman filter prediction is included in the propagation step
            # note: par = [weight, state, EKF_covariance]
            propagated_state = copy.deepcopy(par[1])
            cov_ekf = copy.deepcopy(par[2])

            # Propagate the particle state using the nonlinear process model
            propagated_state[2] += robot_angular_motion
            propagated_state[0] += robot_forward_motion * np.cos(propagated_state[2])
            propagated_state[1] += robot_forward_motion * np.sin(propagated_state[2])
            self.validate_state(propagated_state)

            # Compute Jacobian (df/dx) around current state
            F = np.eye(3)
            F[0,2] = -robot_forward_motion * np.sin(propagated_state[2])
            F[1,2] = robot_forward_motion * np.cos(propagated_state[2])

            # Update covariance
            cov_ekf = F * cov_ekf * np.transpose(F) + self.Q

            # Process measurements one by one (EKF update step)
            for idx, landmark in enumerate(landmarks):

                # Compute expected measurement
                dx = propagated_state[0] - landmark[0]
                dy = propagated_state[1] - landmark[1]
                z1_exp = np.sqrt((dx*dx + dy*dy))
                z2_exp = np.arctan2(dy, dx)

                # Compute Jacobian (dh/dx) around propagated state
                H11 = dx / (np.sqrt(dx*dx + dy*dy))
                H12 = dy / (np.sqrt(dx*dx + dy*dy))
                H21 = 1 / (1 + (dy / dx)**2) * -dy / dx**2
                H22 = 1 / (1 + (dy / dx)**2) * 1 / dx
                H = np.array([[H11, H12, 0], [H21, H22, 0]])

                # Innovation
                y_tilde = np.array([[measurements[idx][0] - z1_exp], [measurements[idx][1] - z2_exp]])
                S = np.dot(np.dot(H, cov_ekf), np.transpose(H)) + self.R

                # Kalman gain
                K = np.dot(np.dot(cov_ekf, np.transpose(H)), linalg.pinv(S))

                # Update state vector and covariance
                delta_state = np.dot(K, y_tilde)
                propagated_state[0] += delta_state[0][0]
                propagated_state[1] += delta_state[1][0]
                propagated_state[2] += delta_state[2][0]
                self.validate_state(propagated_state)
                cov_ekf = np.dot((np.eye(3) - np.dot(K, H)), cov_ekf)

            # New particle state: sample from normal distribution EKF
            updated_state = np.random.multivariate_normal(propagated_state, cov_ekf)
            self.validate_state(updated_state)

            # Compute likelihood using propagated state
            likelihood = self.compute_likelihood(propagated_state, measurements, landmarks)
            # Compute prior (mean is zero vector by default)
            prior = multivariate_normal.pdf(updated_state-propagated_state, cov=self.Q)
            # Importance density
            importance_density = multivariate_normal.pdf(updated_state-propagated_state, cov=cov_ekf)
            # Compute current particle's weight
            weight = likelihood * prior / importance_density
            sum_weights += weight

            # Store updated particle
            new_particles.append([weight, propagated_state, cov_ekf])

        # Normalize particle weight
        if sum_weights < 1e-10:
            print("Warning: sum particles weights very low")
        for par in new_particles:
            par[0] /= sum_weights

        # Update particles
        self.particles = new_particles

        # Resample at each time step
        self.multinomial_resampling()
