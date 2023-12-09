import numpy as np

# Calculate the Gaussian probability
def gaussian_probability(sensor_data, predicted_distance, sensor_noise):
    # print(f"Sensor Data {sensor_data}, Predicted Distance {predicted_distance}, Sensor_Noise : {sensor_noise}")
    exponent = -((sensor_data - predicted_distance) ** 2) / (2 * sensor_noise ** 2)
    probability = (1 / (sensor_noise * np.sqrt(2 * np.pi))) * np.exp(exponent)
    return probability

# Calculate the distance from a particle to the nearest point of an obstacle
def calculate_distance_to_obstacle(position, obstacle):
    px, py = position

    closest_x = max(obstacle.x_min, min(px, obstacle.x_max))
    closest_y = max(obstacle.y_min, min(py, obstacle.y_max))

    distance = np.hypot(px - closest_x, py - closest_y)
    return distance
