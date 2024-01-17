import pygame
import numpy as np
import time

from obstacle import Obstacle, detect_collision
from simulation import initialize_particles, move_particles, calculate_sensor_data, update_weights, estimate_robot_position, resample_particles

# Particle Filter Robot Localization Simulation

# This file is part of a simulation project for robot localization using the Particle Filter
# technique. The simulation employs Pygame for visualization and represents the localization
# process of a robot within an environment containing obstacles.

# - Initialization of Pygame for visualization.
# - Defining the environment size and the number of particles.
# - Creating a list of obstacles within the environment.
# - Handling particle initialization, movement, sensor data, weight updates, and resampling.
# - Estimating the robot's position based on particle weights.
# - Visualizing the robot, particles, and obstacles on the screen.

# The simulation allows the user to control the movement of the robot using keyboard inputs.
# The robot's position is estimated probabilistically using a particle filter.

# Author: Bentchakal Kilyan

def main():
    # Initialize the Pygame library
    pygame.init()
    # Set up the display window
    screen = pygame.display.set_mode((400, 400))
    # Initialize the clock for controlling the frame rate
    clock = pygame.time.Clock()

    # Define the environment size and number of particles
    environment_size = (400, 400)
    num_particles = 500
    # Create a list of obstacles in the environment
    obstacles = [
        Obstacle(50, 50, 120, 80),    
        Obstacle(300, 50, 350, 150), 
        Obstacle(150, 100, 250, 130),
        Obstacle(100, 200, 150, 350), 
        Obstacle(250, 250, 320, 280), 
        Obstacle(50, 300, 120, 350),  
        Obstacle(300, 320, 380, 370) 
    ]
    sensor_noise = 1

    # Initialize particles in the environment
    particle_positions, particle_weights = initialize_particles(num_particles, environment_size)

    # Draw the obstacles on the screen
    for obstacle in obstacles:
        pygame.draw.rect(screen, (0, 0, 255), (obstacle.x_min, obstacle.y_min, obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min))

    # Draw the particles on the screen
    for position in particle_positions:
        pygame.draw.circle(screen, (255, 0, 0), (int(position[0]), int(position[1])), 2)

    # Update the display
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle key presses for particle movement
            keys = pygame.key.get_pressed()
            move_x, move_y = 0, 0
            if keys[pygame.K_LEFT]:
                move_x = -1
            if keys[pygame.K_RIGHT]:
                move_x = 1
            if keys[pygame.K_UP]:
                move_y = -1
            if keys[pygame.K_DOWN]:
                move_y = 1

        # Estimate the robot's position based on particle weights
        robot_x, robot_y = estimate_robot_position(particle_positions, particle_weights)

        # Check for collisions and update particle positions if there is no collision
        if not detect_collision(robot_x, robot_y, move_x, move_y, obstacles, environment_size):
            particle_positions = move_particles(particle_positions, move_x, move_y, environment_size)

        # Update sensor data and particle weights based on the new positions
        sensor_data = calculate_sensor_data(particle_positions, obstacles, sensor_noise)
        particle_weights = update_weights(particle_positions, particle_weights, sensor_data, sensor_noise, obstacles)

        # Resample the particles based on their updated weights
        particle_positions, particle_weights = resample_particles(particle_positions, particle_weights)

        # Clear the screen for the new frame
        screen.fill((0, 0, 0))

        # Draw obstacles
        for obstacle in obstacles:
            pygame.draw.rect(screen, (0, 0, 255), (obstacle.x_min, obstacle.y_min, obstacle.x_max - obstacle.x_min, obstacle.y_max - obstacle.y_min))

        # Draw particles
        for position in particle_positions:
            pygame.draw.circle(screen, (255, 0, 0), (int(position[0]), int(position[1])), 2)

        # Draw the estimated position of the robot
        pygame.draw.circle(screen, (0, 255, 0), (int(robot_x), int(robot_y)), 5)

        # Update the display and control the frame rate
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()