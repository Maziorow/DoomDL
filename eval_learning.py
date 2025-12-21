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

SCREEN_W, SCREEN_H = 120, 90
SCREEN_CHANNELS = 3
SCREEN_SIZE = SCREEN_W * SCREEN_H * SCREEN_CHANNELS

class MinimapViz:
    def __init__(self, width=500, height=500, scale=15):
        self.w = width
        self.h = height
        self.scale = scale
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.path = [] 
        self.off_x = 1500
        self.off_y = 3500 

    def update(self, game_vars):
        if len(game_vars) < 4: return self.canvas
        px, py = game_vars[2], game_vars[3]
        sx = int((px + self.off_x) / self.scale)
        sy = int(self.h - (py + self.off_y) / self.scale)
        self.path.append((sx, sy))
        if len(self.path) > 1:
            cv2.line(self.canvas, self.path[-2], self.path[-1], (255, 255, 255), 1)
        display_img = self.canvas.copy()
        cv2.circle(display_img, (sx, sy), 3, (0, 0, 255), -1)
        if len(self.path) > 0:
            cv2.circle(display_img, self.path[0], 2, (0, 255, 0), -1)
        return display_img

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

    def step(self, action_index):
        actions = [False] * self.action_size
        actions[action_index] = True
        
        reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()
        
        if not done:
            state = self.game.get_state()
            obs = flatten_observation(state.screen_buffer, state.game_variables)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        obs = flatten_observation(state.screen_buffer, state.game_variables)
        return obs, {}

    def close(self):
        self.game.close()

if __name__ == "__main__":
    model_path = "logs/best_model.zip"
    config_path = "env_configurations/doom_min.cfg"

    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        print("Make sure you ran the training script and it saved the model.")
        exit()

    print(f"Loading {model_path}...")
    
    env = VizDoomGym(config_path)
    model = PPO.load(model_path)
    minimap = MinimapViz()

    episodes = 5
    for ep in range(episodes):
        print(f"--- Episode {ep+1} ---")
        obs, _ = env.reset()
        done = False
        
        minimap.path = []
        minimap.canvas[:] = 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            
            obs, reward, done, _, _ = env.step(int(action))

            state = env.game.get_state()
            if state:
                if len(state.game_variables) >= 4:
                    map_img = minimap.update(state.game_variables)
                    cv2.imshow("Minimap", map_img)
                    game_screen = obs[env.num_vars:].reshape(SCREEN_H, SCREEN_W, SCREEN_CHANNELS)
                    cv2.imshow("Game Screen", game_screen)            

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if done:
            print("Episode finished.")

    env.close()
    cv2.destroyAllWindows()