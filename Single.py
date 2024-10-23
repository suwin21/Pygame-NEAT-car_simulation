import pygame
import os
import math
import sys
import neat

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
        self.leave_trail()  # Leave the trail behind
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
        # Draw a trail where the car passes, using white color for the trail
        pygame.draw.circle(TRAIL_SURFACE, (255, 255, 255), self.rect.center, 3)

    def get_data(self):
        input_data = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input_data[i] = radar[1]
        return input_data


def eval_genomes(genomes, config):
    global TRAIL_SURFACE

    # Reset trail surface before each generation
    TRAIL_SURFACE.fill((0, 0, 0, 0))  # Reset the trail surface for each generation

    # Loop through all genomes and their respective IDs
    for genome_id, genome in genomes:
        # Create the car and the neural network for each genome
        car = pygame.sprite.GroupSingle(Car())
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0  # Initialize fitness

        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            SCREEN.blit(TRACK, (0, 0))
            SCREEN.blit(TRAIL_SURFACE, (0, 0))  # Draw the persistent trail surface

            # If the car is dead, end the loop
            if not car.sprite.alive:
                run = False
                break

            # Get the neural network output and update the car's direction
            output = net.activate(car.sprite.get_data())
            if output[0] > 0.7:
                car.sprite.direction = 1  # Turn left
            elif output[1] > 0.7:
                car.sprite.direction = -1  # Turn right
            else:
                car.sprite.direction = 0  # Go straight

            # Update car, calculate fitness, and render the environment
            car.sprite.update()
            car.draw(SCREEN)
            pygame.display.update()

            # Check if the car is driving on its own trail
            if TRAIL_SURFACE.get_at(car.sprite.rect.center) == pygame.Color(255, 255, 255, 255):
                genome.fitness += 2  # Reward for driving on the trail
            else:
                genome.fitness += 1  # Increase fitness for survival

# Setup NEAT Neural Network
def run(config_path):
    # Load the configuration for NEAT
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # Create a population based on the configuration
    pop = neat.Population(config)

    # Add reporters to display progress in the terminal
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # Run NEAT's algorithm for 50 generations, evaluating multiple genomes at a time
    pop.run(eval_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
