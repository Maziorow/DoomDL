import pickle
import numpy as np
import cv2
import vizdoom as vzd
import gymnasium as gym
from gymnasium import spaces
import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
import os
import math
import argparse
import csv
from typing import Callable
import matplotlib.pyplot as plt

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

SCREEN_W, SCREEN_H = 60, 45
SCREEN_CHANNELS = 3
SCREEN_SIZE = SCREEN_W * SCREEN_H * SCREEN_CHANNELS
N_ENVS = 8

REWARD_SCALING = 0.001

class DetailedRewardLogger(BaseCallback):
    def __init__(self, log_file="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.file_handler = open(self.log_file, "w", newline="")
        self.writer = csv.writer(self.file_handler)
        self.writer.writerow([
            "episode", "total_reward", "stuck_penalty", "kill_reward", 
            "hit_reward", "move_reward", "miss_penalty", "damage_taken_penalty"
        ])
        self.episode_count = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_stats" in info:
                self.episode_count += 1
                stats = info["episode_stats"]
                self.writer.writerow([
                    self.episode_count,
                    stats["total_reward"],
                    stats["stuck_penalty"],
                    stats["kill_reward"],
                    stats["hit_reward"],
                    stats["move_reward"],
                    stats["miss_penalty"],
                    stats["damage_taken_penalty"]
                ])
                if self.episode_count % 10 == 0:
                    self.file_handler.flush()
        return True

    def _on_training_end(self):
        self.file_handler.close()

def plot_training_results(log_file="training_log.csv", output_img="training_plot.png"):
    try:
        data = np.genfromtxt(log_file, delimiter=',', names=True)
        
        if data.size == 0:
            print("Log file is empty, skipping plot.")
            return

        episodes = data['episode']
        
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 1, 1)
        plt.plot(episodes, data['total_reward'], label='Total Reward', color='black')
        plt.title("Total Reward per Episode")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 2)
        plt.plot(episodes, data['stuck_penalty'], label='Stuck Penalty', color='red', alpha=0.7)
        plt.plot(episodes, data['damage_taken_penalty'], label='Damage Taken', color='orange', alpha=0.7)
        plt.plot(episodes, data['miss_penalty'], label='Miss Penalty', color='brown', alpha=0.7)
        plt.legend()
        plt.title("Penalties (Negative Reward)")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(episodes, data['kill_reward'], label='Kill Reward', color='green', alpha=0.7)
        plt.plot(episodes, data['hit_reward'], label='Hit Reward', color='blue', alpha=0.7)
        plt.plot(episodes, data['move_reward'], label='Move Reward', color='purple', alpha=0.7)
        plt.legend()
        plt.title("Positive Actions (Bonuses)")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_img)
        print(f"Plot saved to {output_img}")
        plt.close()
    except Exception as e:
        print(f"Error plotting data: {e}")

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
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        with th.no_grad():
            dummy_img = th.zeros(1, SCREEN_CHANNELS, SCREEN_H, SCREEN_W)
            n_flatten = self.cnn(dummy_img).view(1, -1).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + self.n_vars, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        vars_part = observations[:, : self.n_vars]

        screen_part_flat = observations[:, self.n_vars :]
        screen_part_img = screen_part_flat.view(-1, SCREEN_CHANNELS, SCREEN_H, SCREEN_W)

        cnn_out = self.cnn(screen_part_img)
        cnn_out = th.flatten(cnn_out, start_dim=1)

        combined = th.cat((cnn_out, vars_part), dim=1)
        return self.linear(combined)


def flatten_observation(screen, game_vars):
    resized = cv2.resize(screen, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
    norm_screen = resized.astype(np.float32) / 255.0
    trans_screen = np.transpose(norm_screen, (2, 0, 1))
    flat_screen = trans_screen.flatten()
    flat_vars = np.array(game_vars, dtype=np.float32)

    if flat_vars.shape[0] > 0:
        flat_vars[0] /= 100.0  # health
    if flat_vars.shape[0] > 1:
        flat_vars[1] /= 100.0  # ammo
    if flat_vars.shape[0] > 2:
        flat_vars[2] /= 1024.0  # x
    if flat_vars.shape[0] > 3:
        flat_vars[3] /= 1024.0  # y
    if flat_vars.shape[0] > 4:
        flat_vars[4:] /= 50.0

    return np.concatenate([flat_vars, flat_screen])


class VizDoomGym(gym.Env):
    def __init__(self, config_path="env_configurations/doom_min.cfg"):
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(False)
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

        self.c_x_buff = []
        self.c_y_buff = []
        self.MAX_BUFF_SIZE = 10

        # --- REWARD TUNING ---
        # We can set rewards high as at the end they're getting scaled by 1000
        self.GOAL_REWARD = 5000.0
        self.KILL_REWARD = 1000.0
        self.HIT_REWARD = 200.0
        self.ITEM_REWARD = 20.0
        self.MISSED_SHOT_PENALTY = -100.0
        self.WALL_STUCK_PENALTY = -1000.0
        self.MIN_DISTANCE_BEFORE_PENALTY = 0.5
        self.HIT_TAKEN_PENALTY = -500.0
        self.MAP_CELL_SIZE = 64.0

        self.REWARD_SCALING = 0.001

    def _reset_state_tracking(self):
        self.visited_cells = {}
        self.last_cell = None
        self.stuck_timer = 0
        self.last_pos = (0, 0)

        self.episode_hist = {
            "total_reward": 0.0,
            "stuck_penalty": 0.0,
            "kill_reward": 0.0,
            "hit_reward": 0.0,
            "move_reward": 0.0,
            "miss_penalty": 0.0,
            "damage_taken_penalty": 0.0
        }

        state = self.game.get_state()
        if state:
            v = state.game_variables
            self.last_health = v[0]
            self.last_ammo = v[1]
            self.last_pos_x = v[2]
            self.last_pos_y = v[3]
            self.last_hit_count = v[4] if len(v) > 4 else 0
            self.last_item_count = v[5] if len(v) > 5 else 0
            self.last_secret_count = v[6] if len(v) > 6 else 0
            self.last_kill_count = v[7] if len(v) > 7 else 0
        else:
            self.last_health = 100
            self.last_ammo = 50
            self.last_pos_x = 0
            self.last_pos_y = 0
            self.last_hit_count = 0
            self.last_item_count = 0
            self.last_secret_count = 0
            self.last_kill_count = 0

    def actualize_buff(self, c_x, c_y):
        if len(self.c_x_buff) == 0:
            self.c_x_buff = [c_x] * (self.MAX_BUFF_SIZE - 1)
            self.c_y_buff = [c_y] * (self.MAX_BUFF_SIZE - 1)

        if len(self.c_x_buff) >= self.MAX_BUFF_SIZE:
            self.c_x_buff.pop(0)
            self.c_y_buff.pop(0)

        self.c_x_buff.append(c_x)
        self.c_y_buff.append(c_y)

    def _calculate_custom_rewards(self, current_vars):
        reward = 0.0
        
        c_health = current_vars[0]
        c_ammo = current_vars[1]
        c_x, c_y = current_vars[2], current_vars[3]
        c_hits = current_vars[4] if len(current_vars) > 4 else 0
        c_items = current_vars[5] if len(current_vars) > 5 else 0
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
            self.episode_hist["stuck_penalty"] += self.WALL_STUCK_PENALTY * self.REWARD_SCALING # <--- LOG
            self.stuck_timer = 10

        ammo_used = self.last_ammo - c_ammo
        damage_dealt = c_hits - self.last_hit_count

        if ammo_used > 0:
            if damage_dealt <= 0:
                reward += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING
                self.episode_hist["miss_penalty"] += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING # <--- LOG

        if c_kills > self.last_kill_count:
            r = (c_kills - self.last_kill_count) * self.KILL_REWARD
            reward += r * self.REWARD_SCALING
            self.episode_hist["kill_reward"] += r * self.REWARD_SCALING # <--- LOG

        if damage_dealt > 0:
            r = damage_dealt * self.HIT_REWARD
            reward += r * self.REWARD_SCALING
            self.episode_hist["hit_reward"] += r * self.REWARD_SCALING # <--- LOG

        if c_health < self.last_health:
            reward += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING
            self.episode_hist["damage_taken_penalty"] += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING # <--- LOG


        cell_x = int(c_x / self.MAP_CELL_SIZE)
        cell_y = int(c_y / self.MAP_CELL_SIZE)
        current_cell = (cell_x, cell_y)
        if current_cell != self.last_cell:
            if current_cell not in self.visited_cells:
                self.visited_cells[current_cell] = True
                reward += 20.0 * self.REWARD_SCALING
                self.episode_hist["move_reward"] += 20.0 * self.REWARD_SCALING # <--- LOG
            self.last_cell = current_cell

        self.last_health = c_health
        self.last_ammo = c_ammo
        self.last_pos_x = c_x
        self.last_pos_y = c_y
        self.last_hit_count = c_hits
        self.last_kill_count = c_kills
        self.last_item_count = c_items

        return reward

    def step(self, action_index):
        actions = [False] * self.action_size
        actions[action_index] = True

        base_reward = self.game.make_action(actions)
        done = self.game.is_episode_finished()

        info = {}

        if not done:
            state = self.game.get_state()
            obs = flatten_observation(state.screen_buffer, state.game_variables)
            custom_reward = self._calculate_custom_rewards(state.game_variables)
            total_reward = (base_reward * self.REWARD_SCALING)  + custom_reward
            
            self.episode_hist["total_reward"] += total_reward
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            total_reward = base_reward
            self.episode_hist["total_reward"] += base_reward

            if self.last_health > 0:
                total_reward += self.GOAL_REWARD
                self.episode_hist["total_reward"] += self.GOAL_REWARD

            info["episode_stats"] = self.episode_hist

        return obs, total_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self._reset_state_tracking()
        state = self.game.get_state()
        obs = flatten_observation(state.screen_buffer, state.game_variables)
        return obs, {}

    def close(self):
        self.game.close()


def load_expert_data_flat(path, env_num_vars):
    print(f"Loading data from {path}...")
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    obs_list, acts_list = [], []
    for obs, action, reward, next_obs in raw_data:
        game_vars = np.array(obs["gamevariables"], dtype=np.float32)
        if game_vars.shape[0] < env_num_vars:
            pad = np.zeros(env_num_vars - game_vars.shape[0], dtype=np.float32)
            game_vars = np.concatenate([game_vars, pad])
        elif game_vars.shape[0] > env_num_vars:
            game_vars = game_vars[:env_num_vars]
        flat_obs = flatten_observation(obs["screen"], game_vars)
        obs_list.append(flat_obs)
        acts_list.append(action)

    transitions = Transitions(
        obs=np.stack(obs_list),
        acts=np.array(acts_list, dtype=np.int64),
        infos=[{}] * len(acts_list),
        next_obs=np.zeros_like(np.stack(obs_list)),
        dones=np.zeros(len(acts_list), dtype=bool),
    )
    return transitions


def make_env():
    return VizDoomGym()


def main(args):
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    expert_dir = "./doom_expert"
    expert_set = os.listdir(expert_dir)
    model_save_path = "doom_agent_model"
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Creating {N_ENVS} parallel environments...")
    venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    temp = make_env()
    env_vars = temp.num_vars
    temp.close()

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(
            features_extractor_class=FusedInputExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        ),
        verbose=1,
        learning_rate=linear_schedule(args.learning_rate),
        batch_size=256,
        n_steps=2048,
        ent_coef=0.01,
    )
    if args.model != "NONE":
        temp_model = PPO.load(args.model)
        model.set_parameters(temp_model.get_parameters())

    log_csv_path = os.path.join(log_dir, "detailed_rewards.csv")
    reward_logger = DetailedRewardLogger(log_file=log_csv_path)

    policy = model.policy
    for param in policy.features_extractor.parameters():
        param.requires_grad = True
    if args.bc_train:
        for expert_gameplay in expert_set:
            try:
                transitions = load_expert_data_flat(f"./{expert_dir}/{expert_gameplay}", env_vars)
            except Exception as e:
                print(f"Error: {e}")
                return
            print("--- Szkolenie na podstawie nagranej rozgrywki---")

            bc_trainer = BC(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
                policy=policy,
                demonstrations=transitions,
                rng=np.random.default_rng(args.seed),
            )
            bc_trainer.train(n_epochs=args.bc_epochs)

    print("--- Szkolenie ---")

    if args.ppo_train:
        eval_env = make_vec_env(make_env, n_envs=1)

        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=args.eval_frequency,
            deterministic=True,
            render=False,
        )

        callbacks = [eval_callback, reward_logger]

        model.learn(total_timesteps=args.ppo_timesteps, callback=callbacks)

    model.save(model_save_path)
    log_png = os.path.join(log_dir, "training_summary.png")
    plot_training_results(log_file=log_csv_path, output_img=log_png)
    print("Skonczony trening i zapisany model.")
    venv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_train", action="store_true")
    parser.add_argument("--ppo_train", action="store_true")
    parser.add_argument("--bc_epochs", type=int, default=10)
    parser.add_argument("--ppo_timesteps", type=int, default=3000000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--model", type=str, default="NONE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--eval_frequency", type=int, default=20000)

    args = parser.parse_args()
    N_ENVS = args.envs

    main(args)
