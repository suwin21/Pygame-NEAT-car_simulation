import pygame
import os
import math
import sys
import random
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam

# Screen setup (scaled down to 50% of original size)
SCREEN_WIDTH = 622
SCREEN_HEIGHT = 508
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))
TRACK = pygame.transform.scale(TRACK, (SCREEN_WIDTH, SCREEN_HEIGHT))  # Scale track image

# Create a surface to store the trail
TRAIL_SURFACE = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
TRAIL_SURFACE.set_colorkey((0, 0, 0))  # Set black as transparent

# Car class with trail colors for exploration and exploitation
class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.original_image = pygame.transform.scale(self.original_image, (40, 20))  # Scale down car image
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(245, 410))  # Adjust starting position for the new scale
        self.vel_vector = pygame.math.Vector2(0.4, 0)  # Adjust velocity vector for scaling
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []
        self.is_exploring = False  # Track whether the agent is exploring or exploiting

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        self.leave_trail()  # Leave the trail behind
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def drive(self):
        self.rect.center += self.vel_vector * 3  # Adjust car speed for smaller screen

    def collision(self):
        length = 20  # Reduced collision detection length
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        # Draw collision points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 2)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 2)

    def rotate(self):
        # Check the radar data to adjust turning speed when near corners
        if self.radars:
            left_radar = self.radars[0][1]  # Leftmost radar distance
            right_radar = self.radars[-1][1]  # Rightmost radar distance
            
            # Detect when turning is needed by comparing radar distances
            if left_radar < 50:  # Turning right (close to left boundary)
                self.rotation_vel = 10  # Increase rotation speed for sharper turns
                self.angle += self.rotation_vel
                self.vel_vector.rotate_ip(-self.rotation_vel)
            elif right_radar < 50:  # Turning left (close to right boundary)
                self.rotation_vel = 10  # Increase rotation speed for sharper turns
                self.angle -= self.rotation_vel
                self.vel_vector.rotate_ip(self.rotation_vel)
            else:
                self.rotation_vel = 5  # Default turning speed

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def radar(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 100:  # Adjust radar length
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 2)  # Adjust radar point size

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def leave_trail(self):
        # Change trail color based on whether exploring or exploiting
        trail_color = (255, 0, 0) if self.is_exploring else (0, 0, 255)  # Red for exploration, Blue for exploitation
        pygame.draw.circle(TRAIL_SURFACE, trail_color, self.rect.center, 2)  # Adjust trail size

    def get_data(self):
        input_data = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input_data[i] = radar[1]
        return input_data


# PPO agent
class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.lamda = 0.95
        self.learning_rate = 0.0003
        self.entropy_coef = 0.01
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='softmax')  # Output probabilities
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        probabilities = self.model.predict(state, verbose=0)[0]
        action = np.random.choice(self.action_size, p=probabilities)
        return action, probabilities

    def update(self, states, actions, advantages, returns):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            action_prob = tf.reduce_sum(actions * logits, axis=1)
            entropy = -tf.reduce_mean(logits * tf.math.log(logits + 1e-10))  # Entropy term
            loss = -tf.reduce_mean(action_prob * advantages) + self.entropy_coef * entropy  # PPO Loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))


def eval_drl(agent, state_size):
    global TRAIL_SURFACE
    car = pygame.sprite.GroupSingle(Car())

    state = np.reshape(car.sprite.get_data(), [1, state_size])

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))
        SCREEN.blit(TRAIL_SURFACE, (0, 0))

        action, _ = agent.act(state)
        car.sprite.is_exploring = False  # PPO generally exploits learned policy

        if action == 0:
            car.sprite.direction = 1  # Turn left
        elif action == 1:
            car.sprite.direction = -1  # Turn right
        else:
            car.sprite.direction = 0  # Go straight

        car.sprite.update()
        car.draw(SCREEN)
        pygame.display.update()

        next_state = np.reshape(car.sprite.get_data(), [1, state_size])
        reward = 1 if car.sprite.alive else -10
        done = not car.sprite.alive

        state = next_state

        if done:
            break


def run():
    state_size = 5  # Number of radar points
    action_size = 3  # Actions: left, right, straight
    agent = PPOAgent(state_size, action_size)

    episodes = 1000
    for e in range(episodes):
        eval_drl(agent, state_size)


if __name__ == '__main__':
    run()
