import numpy as np
import cv2
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import math
import argparse

SCREEN_W, SCREEN_H = 120, 90
SCREEN_CHANNELS = 3
SCREEN_SIZE = SCREEN_W * SCREEN_H * SCREEN_CHANNELS

class DoomMiniMap:
    def __init__(self):
        self.MAP_RESOLUTION = 64.0 
        self.MAP_BOUNDS = (-3000, 4000, -3000, 4000) 
        
        self.MAP_W = int((self.MAP_BOUNDS[1] - self.MAP_BOUNDS[0]) / self.MAP_RESOLUTION) + 1
        self.MAP_H = int((self.MAP_BOUNDS[3] - self.MAP_BOUNDS[2]) / self.MAP_RESOLUTION) + 1
        
        self.visited_grid = np.zeros((self.MAP_H, self.MAP_W), dtype=np.uint8)
        self.last_cell = None
        
        self.VIEW_CELLS = 40

    def update(self, game_vars):
        if len(game_vars) < 4:
            return np.zeros((500, 500, 3), dtype=np.uint8), False

        px = game_vars[2]
        py = game_vars[3]
        
        idx_x = int((px - self.MAP_BOUNDS[0]) / self.MAP_RESOLUTION)
        idx_y = int((py - self.MAP_BOUNDS[2]) / self.MAP_RESOLUTION)
        
        idx_x = max(0, min(self.MAP_W - 1, idx_x))
        idx_y = max(0, min(self.MAP_H - 1, idx_y))
        
        current_cell = (idx_x, idx_y)
        is_new_cell = False

        if current_cell != self.last_cell:
            if self.visited_grid[idx_y, idx_x] == 0:
                self.visited_grid[idx_y, idx_x] = 255
                is_new_cell = True
            self.last_cell = current_cell

        map_vis = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        map_vis[self.visited_grid == 255] = [0, 255, 0]
        pad = self.VIEW_CELLS
        padded_map = cv2.copyMakeBorder(map_vis, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[50, 50, 50])
        
        center_x = idx_x + pad
        center_y = idx_y + pad
        
        half_view = self.VIEW_CELLS // 2
        x1 = center_x - half_view
        x2 = center_x + half_view
        y1 = center_y - half_view
        y2 = center_y + half_view

        crop = padded_map[y1:y2, x1:x2]
        display_img = cv2.resize(crop, (500, 500), interpolation=cv2.INTER_NEAREST)
        
        center_px = 250 
        cv2.rectangle(display_img, (center_px-5, center_px-5), (center_px+5, center_px+5), (0, 0, 255), -1)

        coord_text = f"X: {px:.1f}, Y: {py:.1f}"
        cv2.putText(display_img, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return display_img, is_new_cell

    def reset(self):
        self.visited_grid = np.zeros((self.MAP_H, self.MAP_W), dtype=np.uint8)
        self.last_cell = None

class RewardCalculator:
    def __init__(self):
        self.GOAL_REWARD = 5000.0
        self.SECRET_REWARD = 2500.0
        self.KILL_REWARD = 1000.0
        self.HIT_REWARD = 200.0
        self.ITEM_REWARD = 20.0
        self.MOVE_REWARD = 60.0 
        self.MISSED_SHOT_PENALTY = -200.0
        self.WALL_STUCK_PENALTY = -50.0
        self.MIN_DISTANCE_BEFORE_PENALTY = 3.5
        self.HIT_TAKEN_PENALTY = -500.0
        self.STAY_ON_VISITED_PENALTY = -15.0
        self.REWARD_SCALING = 0.001
        self.MAX_ACTION_NUMBER = 7000 
        self.MAX_BUFF_SIZE = 10

        self.stuck_timer = 0
        self.c_x_buff = []
        self.c_y_buff = []

        self.last_health = 100
        self.last_ammo = 50
        self.last_hit_count = 0
        self.last_kill_count = 0
        self.last_secret_count = 0

    def actualize_buff(self, c_x, c_y):
        if len(self.c_x_buff) == 0:
            self.c_x_buff = [c_x] * (self.MAX_BUFF_SIZE - 1)
            self.c_y_buff = [c_y] * (self.MAX_BUFF_SIZE - 1)
        if len(self.c_x_buff) >= self.MAX_BUFF_SIZE:
            self.c_x_buff.pop(0)
            self.c_y_buff.pop(0)
        self.c_x_buff.append(c_x)
        self.c_y_buff.append(c_y)

    def reset(self, initial_vars):
        self.stuck_timer = 0
        self.c_x_buff = []
        self.c_y_buff = []
        
        if initial_vars is not None:
            self.last_health = initial_vars[0]
            self.last_ammo = initial_vars[1]
            self.last_hit_count = initial_vars[4] if len(initial_vars) > 4 else 0
            self.last_kill_count = initial_vars[7] if len(initial_vars) > 7 else 0
            self.last_secret_count = initial_vars[6] if len(initial_vars) > 6 else 0
        else:
            self.last_health = 100
            self.last_ammo = 50
            self.last_hit_count = 0
            self.last_kill_count = 0
            self.last_secret_count = 0

    def calculate_step_reward(self, current_vars, base_reward, done, episode_time, is_new_cell):
        if done:
            total_reward = base_reward
            timeout = episode_time >= (self.MAX_ACTION_NUMBER - 1)
            if self.last_health > 0 and not timeout:
                total_reward += self.GOAL_REWARD * self.REWARD_SCALING
            elif self.last_health > 0 and timeout:
                total_reward += -self.GOAL_REWARD * self.REWARD_SCALING
            return total_reward

        reward = 0.0
        c_health = current_vars[0]
        c_ammo = current_vars[1]
        c_x, c_y = current_vars[2], current_vars[3]
        c_hits = current_vars[4] if len(current_vars) > 4 else 0
        c_secrets = current_vars[6] if len(current_vars) > 6 else 0
        c_kills = current_vars[7] if len(current_vars) > 7 else 0

        self.actualize_buff(c_x, c_y)
        dist = math.sqrt((c_x - np.mean(self.c_x_buff)) ** 2 + (c_y - np.mean(self.c_y_buff)) ** 2)
        if dist < self.MIN_DISTANCE_BEFORE_PENALTY:
            self.stuck_timer += 1
        else:
            self.stuck_timer = 0

        if self.stuck_timer > 15:
            reward += self.WALL_STUCK_PENALTY * self.REWARD_SCALING
            self.stuck_timer = 10

        ammo_used = self.last_ammo - c_ammo
        damage_dealt = c_hits - self.last_hit_count

        if ammo_used > 0 and damage_dealt <= 0:
            reward += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING

        if c_kills > self.last_kill_count:
            r = (c_kills - self.last_kill_count) * self.KILL_REWARD
            reward += r * self.REWARD_SCALING

        if c_secrets > self.last_secret_count:
            reward += (self.SECRET_REWARD * self.REWARD_SCALING) * (c_secrets - self.last_secret_count)

        if damage_dealt > 0:
            r = damage_dealt * self.HIT_REWARD
            reward += r * self.REWARD_SCALING

        if c_health < self.last_health:
            reward += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING

        if is_new_cell:
            reward += self.MOVE_REWARD * self.REWARD_SCALING
        else:
            reward += self.STAY_ON_VISITED_PENALTY * self.REWARD_SCALING

        self.last_health = c_health
        self.last_ammo = c_ammo
        self.last_hit_count = c_hits
        self.last_kill_count = c_kills
        self.last_secret_count = c_secrets

        return (base_reward * self.REWARD_SCALING) + reward

class FusedInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        total_input = observation_space.shape[0]
        self.n_vars = total_input - SCREEN_SIZE

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        with th.no_grad():
            dummy_img = th.zeros(1, SCREEN_CHANNELS, SCREEN_H, SCREEN_W)
            n_flatten = self.cnn(dummy_img).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_vars, 1024), 
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        vars_part = observations[:, :self.n_vars]
        screen_part_flat = observations[:, self.n_vars:]
        screen_part_img = screen_part_flat.view(-1, SCREEN_CHANNELS, SCREEN_H, SCREEN_W)
        cnn_out = self.cnn(screen_part_img)
        combined = th.cat((cnn_out, vars_part), dim=1)
        return self.linear(combined)

def flatten_observation(screen, game_vars):
    resized = cv2.resize(screen, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
    norm_screen = resized.astype(np.float32) / 255.0
    trans_screen = np.transpose(norm_screen, (2, 0, 1))
    flat_screen = trans_screen.flatten()
    flat_vars = np.array(game_vars, dtype=np.float32)
    return np.concatenate([flat_vars, flat_screen])


class VizDoomGym(gym.Env):
    def __init__(self, config_path="env_configurations/doom_min.cfg"):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        
        self.game.set_window_visible(True) 
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()

        self.action_size = len(self.game.get_available_buttons())
        self.action_space = spaces.Discrete(self.action_size)

        dummy_state = self.game.get_state()
        self.num_vars = len(dummy_state.game_variables)
        total_size = self.num_vars + SCREEN_SIZE

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )
        
        # Initialize helper classes
        self.minimap = DoomMiniMap()
        self.reward_calc = RewardCalculator()

    def step(self, action_index):
        actions = [False] * self.action_size
        actions[action_index] = True
        
        base_reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()
        episode_time = self.game.get_episode_time()
        
        map_img = None
        calculated_reward = 0.0
        
        if not done:
            state = self.game.get_state()
            game_vars = state.game_variables
            screen = state.screen_buffer
            
            map_img, is_new_cell = self.minimap.update(game_vars)
            
            calculated_reward = self.reward_calc.calculate_step_reward(
                game_vars, base_reward, done, episode_time, is_new_cell
            )
            
            obs = flatten_observation(screen, game_vars)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            calculated_reward = self.reward_calc.calculate_step_reward(
                [], base_reward, done, episode_time, False
            )
            
        info = {"minimap": map_img}
            
        return obs, calculated_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        
        self.minimap.reset()
        
        state = self.game.get_state()
        self.reward_calc.reset(state.game_variables)
        
        obs = flatten_observation(state.screen_buffer, state.game_variables)
        return obs, {}

    def close(self):
        self.game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Doom Agent")
    parser.add_argument("--model-path", type=str, default="logs/best_model.zip", 
                        help="Path to the trained model zip file (default: logs/best_model.zip)")
    args = parser.parse_args()

    model_path = args.model_path
    config_path = "env_configurations/doom_min.cfg""

    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        print("Please check the path or run the training script first.")
        exit()

    print(f"Loading {model_path}...")
    
    env = VizDoomGym(config_path)
    model = PPO.load(model_path)

    episodes = 5
    for ep in range(episodes):
        print(f"--- Episode {ep+1} ---")
        obs, _ = env.reset()
        done = False
        
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, _, info = env.step(int(action))
            total_reward += reward

            if info.get("minimap") is not None:
                cv2.imshow("Advanced Minimap", info["minimap"])
            
            game_screen = obs[env.num_vars:].reshape(SCREEN_H, SCREEN_W, SCREEN_CHANNELS)
            game_screen_big = cv2.resize(game_screen, (480, 360), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Agent View", game_screen_big)            

            if cv2.waitKey(20) & 0xFF == ord('q'):
                done = True
                break
        
        print(f"Episode finished. Total Weighted Reward: {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()