import numpy as np

# Class representing an obstacle in the environment
class Obstacle:
    def __init__(self, x_min, y_min, x_max, y_max):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max

# Check if moving the robot to a new position would result in a collision with any obstacles
def detect_collision(current_x, current_y, move_x, move_y, obstacles, environment_size):
    new_x = current_x + move_x
    new_y = current_y + move_y

    new_x = np.clip(new_x, 0, environment_size[0])
    new_y = np.clip(new_y, 0, environment_size[1])

    for obstacle in obstacles:
        if obstacle.x_min < new_x < obstacle.x_max and obstacle.y_min < new_y < obstacle.y_max:
            return True  

    return False  
