import cv2
import numpy as np
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import time

# ===== PARAMETRY =====
SCREEN_W, SCREEN_H = 60, 45
SCREEN_CHANNELS = 3
SCREEN_SIZE = SCREEN_W * SCREEN_H * SCREEN_CHANNELS
MODEL_PATH = "doom_agent_model.zip"
CONFIG_PATH = "env_configurations/doom_min.cfg"


# ===== FEATURE EXTRACTOR (MUSI BYĆ IDENTYCZNY!) =====
class FusedInputExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        total_input = observation_space.shape[0]
        self.n_vars = total_input - SCREEN_SIZE

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        with th.no_grad():
            dummy = th.zeros(1, 3, SCREEN_H, SCREEN_W)
            n_flatten = self.cnn(dummy).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_vars, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        vars_part = obs[:, : self.n_vars]
        screen_flat = obs[:, self.n_vars :]
        screen = screen_flat.view(-1, 3, SCREEN_H, SCREEN_W)

        cnn_out = self.cnn(screen)
        cnn_out = th.flatten(cnn_out, start_dim=1)

        combined = th.cat((cnn_out, vars_part), dim=1)
        return self.linear(combined)


# ===== OBS FLATTEN =====
def flatten_observation(screen, game_vars):
    screen = cv2.resize(screen, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
    screen = screen.astype(np.float32) / 255.0
    screen = np.transpose(screen, (2, 0, 1)).flatten()
    vars_flat = np.array(game_vars, dtype=np.float32)
    return np.concatenate([vars_flat, screen])


# ===== ENV DO TESTÓW =====
class VizDoomGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(CONFIG_PATH)
        self.game.set_window_visible(True)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.init()

        self.action_size = len(self.game.get_available_buttons())
        self.action_space = spaces.Discrete(self.action_size)

        state = self.game.get_state()
        self.num_vars = len(state.game_variables)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_vars + SCREEN_SIZE,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        s = self.game.get_state()
        obs = flatten_observation(s.screen_buffer, s.game_variables)
        return obs, {}

    def step(self, action):
        actions = [False] * self.action_size
        actions[int(action)] = True

        reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()

        if not done:
            s = self.game.get_state()
            obs = flatten_observation(s.screen_buffer, s.game_variables)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, False, {}

    def close(self):
        self.game.close()


# ===== MAIN TEST =====
def main():
    print("Ładowanie modelu...")
    env = VizDoomGym()

    model = PPO.load(
        MODEL_PATH,
        env=env,
        custom_objects={"features_extractor_class": FusedInputExtractor},
    )

    episodes = 5

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        score = 0.0

        print(f"=== Episode {ep+1} ===")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            score += reward

            if cv2.waitKey(1) == 27:
                break

        print(f"Episode {ep+1} score: {score:.1f}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
