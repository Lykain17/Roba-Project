import numpy as np
from utils import probability, calculate_distance_to_obstacle

# Monte Carlo Localization (MCL)

# This file contains functions for Monte Carlo Localization (MCL). MCL employs a set of
# particles to represent potential robot positions and updates them based on robot motion
# and sensor data, enabling probabilistic localization.

# - Initializing particles with random positions and equal weights.
# - Moving particles based on specified robot movement, with added noise for randomness.
# - Calculating sensor data based on particle positions and obstacles.
# - Predicting the distance from a particle to the nearest obstacle.
# - Updating particle weights using sensor data and predicted distances.
# - Estimating the robot's position based on the weighted average of particle positions.
# - Resampling particles based on their weights.

# Author: Bentchakal Kilyan

# Initialize the particles with random positions and equal weights
def initialize_particles(num_particles, environment_size):
    positions = np.random.uniform(0, environment_size, (num_particles, 2))
    weights = np.ones(num_particles)
    return positions, weights

# Move each particle based on the given movement and add some noise for randomness
def move_particles(positions, move_x, move_y, environment_size, noise=0.5, move_speed=1):
    random_offsets = np.random.normal(0, noise, (len(positions), 2))
    moves = np.array([move_x, move_y]) * move_speed
    positions += moves + random_offsets
    positions = np.clip(positions, [0, 0], environment_size)
    return positions 

# Calculate sensor data based on the particle positions and obstacles
def calculate_sensor_data(positions, obstacles, sensor_noise):
    simulated_sensor_data = []

    for position in positions:
        distances = [calculate_distance_to_obstacle(position, obstacle) for obstacle in obstacles]
        min_distance = min(distances) if distances else float('inf')
        simulated_sensor_data.append(np.random.normal(min_distance, sensor_noise))

    return np.median(simulated_sensor_data)

# Calculate the predicted distance from a particle to the nearest obstacle.
def calculate_predicted_distance(position, obstacles):
    closest_distance = float('inf')

    for obstacle in obstacles:
        distance = calculate_distance_to_obstacle(position, obstacle)
        closest_distance = min(distance, closest_distance)

    return closest_distance

# Update the weights of each particle based on sensor data and predicted distances
def update_weights(positions, weights, sensor_data, sensor_noise, obstacles):
    for i, position in enumerate(positions):
        predicted_distance = calculate_predicted_distance(position, obstacles)
        weight_update = probability(sensor_data, predicted_distance, sensor_noise)
        weights[i] *= weight_update
    return weights

# Estimate the robot's position based on the weighted average of the particle positions.
def estimate_robot_position(positions, weights):
    total_weight = np.sum(weights)
    x = np.sum(positions[:, 0] * weights) / total_weight
    y = np.sum(positions[:, 1] * weights) / total_weight
    return x, y

# Resample the particles based on their weights.
def resample_particles(positions, weights):
    indices = np.random.choice(len(positions), size=len(positions), p=weights/weights.sum())
    new_positions = positions[indices]
    new_weights = np.ones_like(weights)
    return new_positions, new_weights
