##
# Define simulated world
##

# Each landmark has a 2D position (all in meters)
landmark_positions = [[2.0, 2.0], [2.0, 8.0], [9.0, 2.0], [8, 9]]
world_size_x = 10.0
world_size_y = 10.0

##
# True robot properties (simulator settings)
##

# Set point: desired motion robot (displacements)
robot_setpoint_motion_forward = 0.25  # m per time step
robot_setpoint_motion_turn = 0.02     # rad per time step

# True simulated robot motion is set point plus additive zero mean Gaussian noise with these standard deviation
true_robot_motion_forward_std = 0.005  # m
true_robot_motion_turn_std = 0.002     # rad

# Robot measurements are corrupted by measurement noise
true_robot_meas_noise_distance_std = 0.2  # m
true_robot_meas_noise_angle_std = 0.05    # rad

# Initial simulated robot position
robot_initial_x_position = world_size_x * 0.75  # m
robot_initial_y_position = world_size_y * 5.0   # m
robot_initial_heading = 3.14 / 2.0              # rad

##
# Particle filter settings
##
pf_state_limits = [0, world_size_x, 0, world_size_y]

# Process model noise (zero mean additive Gaussian noise)
motion_model_forward_std = 0.10  # m
motion_model_turn_std = 0.02     # rad
process_noise = [motion_model_forward_std, motion_model_turn_std]

# Measurement noise (zero mean additive Gaussian noise)
meas_model_distance_std = 0.4  # m
meas_model_angle_std = 0.3     # rad
measurement_noise = [meas_model_distance_std, meas_model_angle_std]