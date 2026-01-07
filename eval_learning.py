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
from collections import deque

# --- Constants from Training Script ---
SCREEN_W, SCREEN_H = 120, 90
SCREEN_CHANNELS = 3
N_STACK = 4  # Matches training
TOTAL_SCREEN_CHANNELS = SCREEN_CHANNELS * N_STACK

# Map constants matching training
MAP_RESOLUTION = 400
MAP_BOUNDS = (-3000, 4000, -3000, 4000)
MAP_W = int((MAP_BOUNDS[1] - MAP_BOUNDS[0]) / MAP_RESOLUTION) + 1
MAP_H = int((MAP_BOUNDS[3] - MAP_BOUNDS[2]) / MAP_RESOLUTION) + 1
MAP_CHANNELS = 2

SCREEN_FLAT_SIZE = SCREEN_W * SCREEN_H * TOTAL_SCREEN_CHANNELS
MAP_FLAT_SIZE = MAP_W * MAP_H * MAP_CHANNELS
# --------------------------------------

# --- Class definitions required to load the model ---
class ThreeHeadExtractor(BaseFeaturesExtractor):
    """
    Must match the class definition used during training 
    so PPO can load the policy weights correctly.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        total_input = observation_space.shape[0]
        self.n_vars = total_input - SCREEN_FLAT_SIZE - MAP_FLAT_SIZE

        self.screen_cnn = nn.Sequential(
            nn.Conv2d(TOTAL_SCREEN_CHANNELS, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.map_cnn = nn.Sequential(
            nn.Conv2d(MAP_CHANNELS, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        with th.no_grad():
            dummy_screen = th.zeros(1, TOTAL_SCREEN_CHANNELS, SCREEN_H, SCREEN_W)
            dummy_map = th.zeros(1, MAP_CHANNELS, MAP_H, MAP_W)
            screen_out_dim = self.screen_cnn(dummy_screen).shape[1]
            map_out_dim = self.map_cnn(dummy_map).shape[1]

        concat_dim = screen_out_dim + map_out_dim + self.n_vars
        
        self.linear = nn.Sequential(
            nn.Linear(concat_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        vars_part = observations[:, : self.n_vars]
        
        screen_start = self.n_vars
        screen_end = self.n_vars + SCREEN_FLAT_SIZE
        screen_flat = observations[:, screen_start : screen_end]
        screen_img = screen_flat.view(-1, TOTAL_SCREEN_CHANNELS, SCREEN_H, SCREEN_W)

        map_flat = observations[:, screen_end :]
        map_img = map_flat.view(-1, MAP_CHANNELS, MAP_H, MAP_W)

        screen_feat = self.screen_cnn(screen_img)
        map_feat = self.map_cnn(map_img)

        combined = th.cat((screen_feat, map_feat, vars_part), dim=1)
        return self.linear(combined)

class ObservationMiniMap:
    """The logical map used for Agent Input (from training script)"""
    def __init__(self):
        self.state = np.zeros((MAP_CHANNELS, MAP_H, MAP_W), dtype=np.float32)

    def update(self, x, y):
        idx_x = int((x - MAP_BOUNDS[0]) / MAP_RESOLUTION)
        idx_y = int((y - MAP_BOUNDS[2]) / MAP_RESOLUTION)
        idx_x = max(0, min(MAP_W - 1, idx_x))
        idx_y = max(0, min(MAP_H - 1, idx_y))
        
        self.state[0, idx_y, idx_x] = 1.0
        self.state[1] = 0.0
        self.state[1, idx_y, idx_x] = 1.0
        return self.state

    def reset(self):
        self.state = np.zeros((MAP_CHANNELS, MAP_H, MAP_W), dtype=np.float32)
        return self.state

class VisualMiniMap:
    """The visual map used for Human UI (from eval script)"""
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
            return np.zeros((500, 500, 3), dtype=np.uint8)

        px = game_vars[2]
        py = game_vars[3]
        
        idx_x = int((px - self.MAP_BOUNDS[0]) / self.MAP_RESOLUTION)
        idx_y = int((py - self.MAP_BOUNDS[2]) / self.MAP_RESOLUTION)
        idx_x = max(0, min(self.MAP_W - 1, idx_x))
        idx_y = max(0, min(self.MAP_H - 1, idx_y))
        
        current_cell = (idx_x, idx_y)
        if current_cell != self.last_cell:
            if self.visited_grid[idx_y, idx_x] == 0:
                self.visited_grid[idx_y, idx_x] = 255
            self.last_cell = current_cell

        map_vis = np.zeros((self.MAP_H, self.MAP_W, 3), dtype=np.uint8)
        map_vis[self.visited_grid == 255] = [0, 255, 0]
        pad = self.VIEW_CELLS
        padded_map = cv2.copyMakeBorder(map_vis, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[50, 50, 50])
        
        center_x = idx_x + pad
        center_y = idx_y + pad
        half_view = self.VIEW_CELLS // 2
        crop = padded_map[center_y - half_view : center_y + half_view, center_x - half_view : center_x + half_view]
        display_img = cv2.resize(crop, (500, 500), interpolation=cv2.INTER_NEAREST)
        
        center_px = 250 
        cv2.rectangle(display_img, (center_px-5, center_px-5), (center_px+5, center_px+5), (0, 0, 255), -1)
        coord_text = f"X: {px:.1f}, Y: {py:.1f}"
        cv2.putText(display_img, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return display_img

    def reset(self):
        self.visited_grid = np.zeros((self.MAP_H, self.MAP_W), dtype=np.uint8)
        self.last_cell = None

def flatten_observation(screen_stack, game_vars, minimap_state):
    """
    Replicates the logic from train_from_saved_trajectory.py
    Combines N_STACK frames + Game Vars + MiniMap
    """
    processed_frames = []
    for frame in screen_stack:
        resized = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        trans = np.transpose(norm, (2, 0, 1))
        processed_frames.append(trans)
    
    stacked_screen = np.concatenate(processed_frames, axis=0)
    flat_screen = stacked_screen.flatten()

    # Normalize vars as done in training
    flat_vars = np.array(game_vars, dtype=np.float32)
    if flat_vars.shape[0] > 0: flat_vars[0] /= 100.0 # Health
    if flat_vars.shape[0] > 1: flat_vars[1] /= 100.0 # Ammo
    if flat_vars.shape[0] > 2: flat_vars[2] /= 100.0 # Pos X
    if flat_vars.shape[0] > 3: flat_vars[3] /= 100.0 # Pos Y
    if flat_vars.shape[0] > 4: flat_vars[4:] /= 50.0 # Counts

    flat_map = minimap_state.flatten()

    return np.concatenate([flat_vars, flat_screen, flat_map])


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
        
        # Calculate size based on training logic
        total_size = self.num_vars + SCREEN_FLAT_SIZE + MAP_FLAT_SIZE

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )
        
        # Initialize helpers
        self.obs_minimap = ObservationMiniMap() # For Model
        self.vis_minimap = VisualMiniMap()      # For Display
        self.frame_buffer = deque(maxlen=N_STACK)

    def step(self, action_index):
        actions = [False] * self.action_size
        actions[action_index] = True
        
        base_reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()
        
        map_vis_img = None
        
        if not done:
            state = self.game.get_state()
            game_vars = state.game_variables
            
            # 1. Update frame buffer
            self.frame_buffer.append(state.screen_buffer)
            
            # 2. Update Model Map
            map_state_obs = self.obs_minimap.update(game_vars[2], game_vars[3])
            
            # 3. Update Visual Map
            map_vis_img = self.vis_minimap.update(game_vars)
            
            # 4. Create correct observation
            obs = flatten_observation(self.frame_buffer, game_vars, map_state_obs)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            
        info = {"minimap": map_vis_img}
        return obs, base_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        
        self.obs_minimap.reset()
        self.vis_minimap.reset()
        self.frame_buffer.clear()
        
        state = self.game.get_state()
        
        # Fill buffer with initial frame
        for _ in range(N_STACK):
            self.frame_buffer.append(state.screen_buffer)
            
        map_state_obs = self.obs_minimap.update(state.game_variables[2], state.game_variables[3])
        
        obs = flatten_observation(self.frame_buffer, state.game_variables, map_state_obs)
        return obs, {}

    def close(self):
        self.game.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Doom Agent")
    parser.add_argument("--model-path", type=str, default="logs/best_model.zip", 
                        help="Path to the trained model zip file")
    args = parser.parse_args()

    model_path = args.model_path
    config_path = "env_configurations/doom_min.cfg"

    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        print("Please check the path or run the training script first.")
        exit()

    print(f"Loading {model_path}...")
    
    env = VizDoomGym(config_path)
    
    # We must ensure custom objects are available, though usually SB3 handles it 
    # if the class matches what is in the save file.
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
            
            # Extract current frame from the observation for display
            # We skip the vars (env.num_vars) and take the last frame from the stack
            # Logic: Vars -> Frame1 -> Frame2 -> Frame3 -> Frame4 -> Map
            # We want Frame4 (most recent)
            
            # It is easier to just grab the raw buffer from the env object for visualization
            # since extracting it from the flattened 'obs' vector is complex arithmetic.
            if len(env.frame_buffer) > 0:
                raw_screen = env.frame_buffer[-1]
                # VizDoom returns RGB, cv2 expects BGR
                bgr_screen = cv2.cvtColor(raw_screen, cv2.COLOR_RGB2BGR)
                game_screen_big = cv2.resize(bgr_screen, (480, 360), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("Agent View", game_screen_big)            

            if cv2.waitKey(50) & 0xFF == ord('q'): # Slower waitKey to make it watchable
                done = True
                break
        
        print(f"Episode finished. Total Reward: {total_reward:.2f}")

    env.close()
    cv2.destroyAllWindows()