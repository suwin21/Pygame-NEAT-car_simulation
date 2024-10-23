import pygame
import os
import math
import sys
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
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
        if self.direction == 1:
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1:
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.9  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))  # Updated learning_rate
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Exploration vs Exploitation
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size), True  # Exploration
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]), False  # Exploitation

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


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

        action, is_exploring = agent.act(state)
        car.sprite.is_exploring = is_exploring  # Update the trail color

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
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        if done:
            break


# Setup Double DQN Neural Network
def run():
    state_size = 5  # Number of radar points
    action_size = 3  # Actions: left, right, straight
    agent = DQNAgent(state_size, action_size)

    episodes = 1000
    for e in range(episodes):
        eval_drl(agent, state_size)
        agent.replay(32)


if __name__ == '__main__':
    run()
