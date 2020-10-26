from .particle_filter_sir import ParticleFilterSIR


class ParticleFilterMWR(ParticleFilterSIR):
    """
    Notes:
        * State is (x, y, heading), where x and y are in meters and heading in radians
        * State space assumed limited size in each dimension, world is cyclic (hence leaving at x_max means entering at
        x_min)
        * Apply core if the reciprocal of the maximum weight drops below a specific value: max weight core
         (MWR)
    """

    def __init__(self,
                 number_of_particles,
                 limits,
                 process_noise,
                 measurement_noise,
                 resampling_algorithm,
                 resampling_threshold):
        """
        Initialize a particle filter that performs resampling whenever the reciprocal of maximum particle weight
        falls below a user specified threshold value.

        :param number_of_particles: Number of particles.
        :param limits: List with maximum and minimum values for x and y dimension: [xmin, xmax, ymin, ymax].
        :param process_noise: Process noise parameters (standard deviations): [std_forward, std_angular].
        :param measurement_noise: Measurement noise parameters (standard deviations): [std_range, std_angle].
        :param resampling_algorithm: Algorithm that must be used for core.
        :param resampling_threshold: Resample whenever the reciprocal of the maximum particle weight falls below this
        value.
        """
        # Initialize sir particle filter class
        ParticleFilterSIR.__init__(self, number_of_particles, limits, process_noise, measurement_noise, resampling_algorithm)

        # Set MWR specific properties
        self.resampling_threshold = resampling_threshold

    def needs_resampling(self):
        """
        Override method that determines whether or not a core step is needed for the current particle filter state
        estimate. Resampling only occurs if the reciprocal of the maximum particle weight falls below the user-specified
        threshold. The reciprocal of the maximum weight is defined by P_N^2 in [1].

        [1] Martino, Luca, Victor Elvira, and Francisco Louzada. "Effective sample size for importance sampling based on
        discrepancy measures." Signal Processing 131 (2017): 386-401.

        :return: Boolean indicating whether or not core is needed.
        """
        max_weight = 0
        for par in self.particles:
            max_weight = max(max_weight, par[0])

        return 1.0 / max_weight < self.resampling_threshold
