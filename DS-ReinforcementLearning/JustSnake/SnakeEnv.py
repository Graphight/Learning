from gym import spaces
from collections import deque
from tqdm.auto import tqdm

import numpy as np

import gym
import os
import random
import cv2
import time


CELL_SIZE = 10
GRID_SIZE = 500
MAX_DISTANCE = GRID_SIZE * 2
IMG_TITLE = "Snek Game"

APPLE_REWARD = 100_000
SNAKE_LEN_GOAL = 30
SIM_SPEED = 0.01   # Delay in seconds between frames


class SnekEnv(gym.Env):
    
    def __init__(self):
        super(SnekEnv, self).__init__()
        
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Box(
            low=0, 
            high=GRID_SIZE, 
            shape=(6 + SNAKE_LEN_GOAL,),
            dtype=np.int64
        )

    def collision_with_boundaries(self):
        return sum([
            (self.snake_head[0] >= GRID_SIZE),
            (self.snake_head[0] < 0),
            (self.snake_head[1] >= GRID_SIZE),
            (self.snake_head[1] < 0)
        ]) >= 1

    def collision_with_self(self):
        snake_head = self.snake_position[0]
        if snake_head in self.snake_position[1:]:
            return 1
        else:
            return 0

    def generate_apple_position(self):
        return [
            random.randrange(1, GRID_SIZE // CELL_SIZE) * CELL_SIZE,
            random.randrange(1, GRID_SIZE // CELL_SIZE) * CELL_SIZE
        ]

    def generate_reward_background(self):
        rows = []

        for y in range(0, GRID_SIZE):
            row = []
            for x in range(0, GRID_SIZE):
                apple_delta_x = x - (self.apple_position[0] + CELL_SIZE // 2)
                apple_delta_y = y - (self.apple_position[1] + CELL_SIZE // 2)
                distance_man = abs(apple_delta_x) + abs(apple_delta_y)
                norm_distance_man = self.determine_reward(distance_man)
                if norm_distance_man >= 0:
                    row.append([norm_distance_man] * 3)
                else:
                    row.append([-norm_distance_man, 0, 0])
            rows.append(row)

        self.cached_background = np.array(rows)

    def determine_observations(self):
        # determine the observation
        self.head_x = self.snake_head[0]
        self.head_y = self.snake_head[1]
        self.snake_length = len(self.snake_position)
        self.apple_delta_x = self.head_x - (self.apple_position[0] + CELL_SIZE // 2)
        self.apple_delta_y = self.head_y - (self.apple_position[1] + CELL_SIZE // 2)
        self.distance_man = abs(self.apple_delta_x) + abs(self.apple_delta_y)

    def determine_reward(self, distance_man):
        return (GRID_SIZE // 2 - distance_man) / GRID_SIZE
    
    def step(self, action):
        self.reward = 0
        cv2.imshow(IMG_TITLE, self.img)
        cv2.waitKey(1)
        # self.img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype='uint8')
        self.img = self.cached_background.copy()

        # Display Apple
        cv2.rectangle(
            self.img,
            (self.apple_position[0], self.apple_position[1]),
            (self.apple_position[0] + CELL_SIZE, self.apple_position[1] + CELL_SIZE),
            (0, 0, 255),
            3
        )
        
        # Display Snake
        for position in self.snake_position:
            cv2.rectangle(
                self.img,
                (position[0], position[1]),
                (position[0] + CELL_SIZE, position[1] + CELL_SIZE),
                (0, 255, 0),
                3
            )

        # Takes step after fixed time
        t_end = time.time() + SIM_SPEED
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(1)
            else:
                continue
        
        # Don't let the snake turn back on itself
        button_direction = action
        if button_direction == 0 and self.prev_button_direction != 1:
            button_direction = 0
        elif button_direction == 1 and self.prev_button_direction != 0:
            button_direction = 1
        elif button_direction == 3 and self.prev_button_direction != 2:
            button_direction = 3
        elif button_direction == 2 and self.prev_button_direction != 3:
            button_direction = 2
        else:
            button_direction = self.prev_button_direction
        self.prev_button_direction = button_direction
        self.prev_actions.append(button_direction)
        
        # Change the head position based on the button direction
        if button_direction == 1:
            self.snake_head[0] += CELL_SIZE
        elif button_direction == 0:
            self.snake_head[0] -= CELL_SIZE
        elif button_direction == 2:
            self.snake_head[1] += CELL_SIZE
        elif button_direction == 3:
            self.snake_head[1] -= CELL_SIZE
        
        # Increase Snake length on eating apple
        if self.snake_head == self.apple_position:
            self.apple_position = self.generate_apple_position()
            self.snake_position.insert(0, list(self.snake_head))
            self.reward += APPLE_REWARD
            self.generate_reward_background()
        else:
            self.snake_position.insert(0, list(self.snake_head))
            self.snake_position.pop()

        # On collision kill the snake
        if self.collision_with_boundaries() or self.collision_with_self():
            self.done = True
        
        self.determine_observations()
        self.reward += self.determine_reward(self.distance_man)

        if self.done:
            self.reward = (-0.1 * APPLE_REWARD)
        info = {}
                
        # Create observation
        observation = [self.head_x, self.head_y, self.apple_delta_x, self.apple_delta_y, self.snake_length, self.distance_man] + list(self.prev_actions)
        observation = np.array(observation)

        return observation, self.reward, self.done, info
    
    def reset(self):        
        self.reward = 0
        self.same_moves = 0
        
        # Initial Snake and Apple position
        self.snake_position = [
            [GRID_SIZE // 2, GRID_SIZE // 2], 
            [(GRID_SIZE // 2) - CELL_SIZE, GRID_SIZE // 2],
            [(GRID_SIZE // 2) - (CELL_SIZE * 2), GRID_SIZE // 2]
        ]
        self.apple_position = self.generate_apple_position()
        self.prev_button_direction = 1
        self.button_direction = 1
        self.snake_head = [GRID_SIZE // 2, GRID_SIZE // 2]
        self.generate_reward_background()
        self.img = self.cached_background.copy()

        self.prev_reward = 0
        self.done = False
        
        # Determine the observation
        head_x = self.snake_head[0]
        head_y = self.snake_head[1]
        snake_length = len(self.snake_position)
        apple_delta_x = head_x - (self.apple_position[0] + CELL_SIZE // 2)
        apple_delta_y = head_y - (self.apple_position[1] + CELL_SIZE // 2)
        distance_man = abs(apple_delta_x) + abs(apple_delta_y)
        
        self.prev_actions = deque([-1] * SNAKE_LEN_GOAL, maxlen=SNAKE_LEN_GOAL)
                
        # Create observation
        observation = [head_x, head_y, apple_delta_x, apple_delta_y, snake_length, distance_man] + list(self.prev_actions)
        observation = np.array(observation)
        
        return observation

        

def main():
    from stable_baselines3 import PPO
    import os
    import time

    model_name = f"snake_{int(time.time())}"
    models_dir = f"models/{model_name}/"
    logdir = f"logs/{model_name}/"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    env = SnekEnv()
    env.reset()

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    iters = 0
    while iters < 10:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS * iters}")

    cv2.destroyAllWindows()
        
        
if __name__ == "__main__":
    main()