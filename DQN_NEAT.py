import pygame
import os
import math
import sys
import neat
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Screen setup
SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))

# Create a surface to store the trail
TRAIL_SURFACE = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
TRAIL_SURFACE.set_colorkey((0, 0, 0))  # Set the black color as transparent

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820))
        self.vel_vector = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        self.leave_trail()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.radar(radar_angle)
        self.collision()

    def drive(self):
        self.rect.center += self.vel_vector * 6

    def collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        # Draw Collision Points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

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

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, dist])

    def leave_trail(self):
        pygame.draw.circle(TRAIL_SURFACE, (255, 255, 255), self.rect.center, 3)

    def get_data(self):
        return [radar[1] for radar in self.radars]

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

def eval_genomes(genomes, config):
    global TRAIL_SURFACE
    TRAIL_SURFACE.fill((0, 0, 0, 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for genome_id, genome in genomes:
        car = pygame.sprite.GroupSingle(Car())
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        dqn = DQN(5, 3).to(device)
        target_dqn = DQN(5, 3).to(device)
        target_dqn.load_state_dict(dqn.state_dict())
        
        optimizer = optim.Adam(dqn.parameters())
        criterion = nn.MSELoss()
        
        replay_buffer = ReplayBuffer(10000)
        
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01
        
        genome.fitness = 0
        
        for episode in range(100):  # Run for 100 episodes
            state = car.sprite.get_data()
            total_reward = 0
            
            while car.sprite.alive:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                SCREEN.blit(TRACK, (0, 0))
                SCREEN.blit(TRAIL_SURFACE, (0, 0))

                # Epsilon-greedy action selection
                if random.random() < epsilon:
                    action = random.randint(0, 2)
                else:
                    with torch.no_grad():
                        q_values = dqn(torch.FloatTensor(state).unsqueeze(0).to(device))
                        action = q_values.max(1)[1].item()

                # Update car direction based on action
                car.sprite.direction = action - 1  # Map 0, 1, 2 to -1, 0, 1

                # Update car and get new state
                car.sprite.update()
                next_state = car.sprite.get_data()

                # Calculate reward
                reward = 1 if car.sprite.alive else -10
                if TRAIL_SURFACE.get_at(car.sprite.rect.center) == pygame.Color(255, 255, 255, 255):
                    reward += 2

                # Store transition in replay buffer
                replay_buffer.push(state, action, reward, next_state, not car.sprite.alive)

                # Train the DQN
                if len(replay_buffer) > 128:
                    try:
                        batch = replay_buffer.sample(128)
                        state_batch, action_batch, reward_batch, next_state_batch, done_batch = [torch.FloatTensor(x).to(device) for x in batch]

                        q_values = dqn(state_batch).gather(1, action_batch.long().unsqueeze(1))
                        next_q_values = target_dqn(next_state_batch).max(1)[0].detach()
                        expected_q_values = reward_batch + (0.99 * next_q_values * (1 - done_batch))

                        loss = criterion(q_values, expected_q_values.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    except Exception as e:
                        print(f"Error during training: {e}")
                        continue

                state = next_state
                total_reward += reward
                
                car.draw(SCREEN)
                pygame.display.update()

            # Update target network
            if episode % 10 == 0:
                target_dqn.load_state_dict(dqn.state_dict())

            # Decay epsilon
            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            genome.fitness += total_reward

def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)