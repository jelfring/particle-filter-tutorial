from .particle_filter_sir import ParticleFilterSIR


class ParticleFilterNEPR(ParticleFilterSIR):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Apply approximate number of effective particles core (NEPR)
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 number_of_effective_particles_threshold):
        """
        Initialize a particle filter that performs resampling whenever the approximated number of effective particles
        falls below a user specified threshold value.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        :param number_of_effective_particles_threshold: Resample whenever approximate number of effective particles
        falls below this value.
        """
        # Initialize sir particle filter class
        ParticleFilterSIR.__init__(self, number_of_particles, limits, process_noise, measurement_noise, resampling_algorithm)

        # Set NEPR specific properties
        self.resampling_threshold = number_of_effective_particles_threshold

    def needs_resampling(self):
        """
        Override method that determines whether or not a step is needed for the current particle filter state
        estimate. Resampling only occurs if the approximated number of effective particles falls below the
        user-specified threshold. Approximate number of effective particles: 1 / (sum_i^N wi^2), P_N^2 in [1].

        [1] Martino, Luca, Victor Elvira, and Francisco Louzada. "Effective sample size for importance sampling based on
        discrepancy measures." Signal Processing 131 (2017): 386-401.

        :return: Boolean indicating whether or not core is needed.
        """
        #
        sum_weights_squared = 0
        for par in self.particles:
            sum_weights_squared += par[0] * par[0]

        return 1.0 / sum_weights_squared < self.resampling_threshold
