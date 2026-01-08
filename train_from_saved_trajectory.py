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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
import os
import math
import argparse
import csv
from typing import Callable
import matplotlib.pyplot as plt
from collections import deque

SCREEN_W, SCREEN_H = 120, 90
SCREEN_CHANNELS = 3
N_STACK = 4
TOTAL_SCREEN_CHANNELS = SCREEN_CHANNELS * N_STACK

MAP_RESOLUTION = 400
MAP_BOUNDS = (-3000, 4000, -3000, 4000)
MAP_W = int((MAP_BOUNDS[1] - MAP_BOUNDS[0]) / MAP_RESOLUTION) + 1
MAP_H = int((MAP_BOUNDS[3] - MAP_BOUNDS[2]) / MAP_RESOLUTION) + 1
MAP_CHANNELS = 2

SCREEN_FLAT_SIZE = SCREEN_W * SCREEN_H * TOTAL_SCREEN_CHANNELS
MAP_FLAT_SIZE = MAP_W * MAP_H * MAP_CHANNELS

REWARD_SCALING = 0.001

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class DetailedRewardLogger(BaseCallback):
    def __init__(self, log_file="training_log.csv", verbose=0):
        super().__init__(verbose)
        self.log_file = log_file
        self.file_handler = open(self.log_file, "w", newline="")
        self.writer = csv.writer(self.file_handler)
        self.writer.writerow([
            "episode", "total_reward", "stuck_penalty", "kill_reward", 
            "hit_reward", "move_reward", "miss_penalty", "damage_taken_penalty",
            "secret_reward", "visited_penalty", "goal_reward"
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
                    stats["damage_taken_penalty"],
                    stats.get("secret_reward", 0.0),
                    stats.get("visited_penalty", 0.0),
                    stats.get("goal_reward", 0.0)
                ])
                if self.episode_count % 10 == 0:
                    self.file_handler.flush()
        return True

    def _on_training_end(self):
        self.file_handler.close()

def plot_training_results(log_file="training_log.csv", output_img="training_plot.png"):
    try:
        data = np.genfromtxt(log_file, delimiter=',', names=True)
        if data.size == 0: return

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
        if 'visited_penalty' in data.dtype.names:
            plt.plot(episodes, data['visited_penalty'], label='Visited Penalty', color='pink', alpha=0.7)
        plt.legend()
        plt.title("Penalties")
        plt.grid(True, alpha=0.3)

        plt.subplot(3, 1, 3)
        plt.plot(episodes, data['kill_reward'], label='Kill Reward', color='green', alpha=0.7)
        plt.plot(episodes, data['move_reward'], label='Move Reward', color='purple', alpha=0.7)
        if 'secret_reward' in data.dtype.names:
            plt.plot(episodes, data['secret_reward'], label='Secret Reward', color='gold', alpha=0.7)
        if 'goal_reward' in data.dtype.names:
            plt.plot(episodes, data['goal_reward'], label='Goal Reward', color='blue', alpha=0.7)
        plt.legend()
        plt.title("Bonuses")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_img)
        plt.close()
    except Exception as e:
        print(f"Error plotting data: {e}")

class DoomMiniMap:
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

class ThreeHeadExtractor(BaseFeaturesExtractor):
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

def flatten_observation(screen_stack, game_vars, minimap_state):
    processed_frames = []
    for frame in screen_stack:
        resized = cv2.resize(frame, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
        norm = resized.astype(np.float32) / 255.0
        trans = np.transpose(norm, (2, 0, 1))
        processed_frames.append(trans)
    
    stacked_screen = np.concatenate(processed_frames, axis=0)
    flat_screen = stacked_screen.flatten()

    flat_vars = np.array(game_vars, dtype=np.float32)
    if flat_vars.shape[0] > 0: flat_vars[0] /= 100.0
    if flat_vars.shape[0] > 1: flat_vars[1] /= 100.0
    if flat_vars.shape[0] > 2: flat_vars[2] /= 100.0
    if flat_vars.shape[0] > 3: flat_vars[3] /= 100.0
    if flat_vars.shape[0] > 4: flat_vars[4:] /= 50.0

    flat_map = minimap_state.flatten()

    return np.concatenate([flat_vars, flat_screen, flat_map])

class VizDoomGym(gym.Env):
    def __init__(self, config_path="env_configurations/doom_min.cfg"):
        global args
        self.MAX_ACTION_NUMBER = 7000
        super().__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(False)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_episode_timeout(self.MAX_ACTION_NUMBER)
        self.game.init()

        self.action_size = len(self.game.get_available_buttons())
        self.action_space = spaces.Discrete(self.action_size)

        self.minimap = DoomMiniMap()
        self.frame_buffer = deque(maxlen=N_STACK)

        dummy_state = self.game.get_state()
        self.num_vars = len(dummy_state.game_variables)
        total_size = self.num_vars + SCREEN_FLAT_SIZE + MAP_FLAT_SIZE

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

        self.c_x_buff = []
        self.c_y_buff = []
        self.MAX_BUFF_SIZE = 10

        self.GOAL_REWARD = args.goal_reward
        self.SECRET_REWARD = args.secret_reward
        self.KILL_REWARD = args.kill_reward
        self.HIT_REWARD = args.hit_reward
        self.ITEM_REWARD = args.item_reward
        self.MOVE_REWARD = args.move_reward
        self.MISSED_SHOT_PENALTY = args.missed_shot_penalty
        self.WALL_STUCK_PENALTY = args.wall_stuck_penalty
        self.MIN_DISTANCE_BEFORE_PENALTY = args.min_distance_before_penalty
        self.HIT_TAKEN_PENALTY = args.hit_taken_penalty
        self.MAP_CELL_SIZE = args.map_cell_size
        self.REWARD_SCALING = args.reward_scaling
        self.STAY_ON_VISITED_PENALTY = args.stay_on_visited_penalty

    def _reset_state_tracking(self):
        self.visited_cells = {}
        self.last_cell = None
        self.stuck_timer = 0
        
        self.episode_hist = {
            "total_reward": 0.0, "stuck_penalty": 0.0, "kill_reward": 0.0,
            "hit_reward": 0.0, "move_reward": 0.0, "secret_reward": 0.0,
            "miss_penalty": 0.0, "damage_taken_penalty": 0.0,
            "visited_penalty": 0.0, "goal_reward": 0.0
        }
        
        self.minimap.reset()
        self.frame_buffer.clear()

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
            self.last_pos_x = 0; self.last_pos_y = 0
            self.last_hit_count = 0; self.last_item_count = 0
            self.last_secret_count = 0; self.last_kill_count = 0

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
            self.episode_hist["stuck_penalty"] += self.WALL_STUCK_PENALTY * self.REWARD_SCALING
            self.stuck_timer = 10

        ammo_used = self.last_ammo - c_ammo
        damage_dealt = c_hits - self.last_hit_count

        if ammo_used > 0 and damage_dealt <= 0:
            reward += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING
            self.episode_hist["miss_penalty"] += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING

        if c_kills > self.last_kill_count:
            r = (c_kills - self.last_kill_count) * self.KILL_REWARD
            reward += r * self.REWARD_SCALING
            self.episode_hist["kill_reward"] += r * self.REWARD_SCALING

        if c_secrets > self.last_secret_count:
            reward += (self.SECRET_REWARD * self.REWARD_SCALING) * (c_secrets - self.last_secret_count)
            self.episode_hist["secret_reward"] += (self.SECRET_REWARD * self.REWARD_SCALING) * (c_secrets - self.last_secret_count)
            self.last_secret_count = c_secrets

        if damage_dealt > 0:
            r = damage_dealt * self.HIT_REWARD
            reward += r * self.REWARD_SCALING
            self.episode_hist["hit_reward"] += r * self.REWARD_SCALING

        if c_health < self.last_health:
            reward += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING
            self.episode_hist["damage_taken_penalty"] += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING

        cell_x = int(c_x / self.MAP_CELL_SIZE)
        cell_y = int(c_y / self.MAP_CELL_SIZE)
        current_cell = (cell_x, cell_y)
        
        if current_cell != self.last_cell:
            if current_cell not in self.visited_cells:
                self.visited_cells[current_cell] = True
                reward += self.MOVE_REWARD * self.REWARD_SCALING
                self.episode_hist["move_reward"] += self.MOVE_REWARD * self.REWARD_SCALING
        else:
            p = self.STAY_ON_VISITED_PENALTY * self.REWARD_SCALING
            reward += p
            self.episode_hist["visited_penalty"] += p
            
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
        timeout = self.game.get_episode_time() >= (self.MAX_ACTION_NUMBER - 1)
        info = {}

        if not done:
            state = self.game.get_state()
            self.frame_buffer.append(state.screen_buffer)
            map_state = self.minimap.update(state.game_variables[2], state.game_variables[3])
            
            obs = flatten_observation(self.frame_buffer, state.game_variables, map_state)
            
            custom_reward = self._calculate_custom_rewards(state.game_variables)
            total_reward = (base_reward * self.REWARD_SCALING)  + custom_reward
            self.episode_hist["total_reward"] += total_reward
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            total_reward = base_reward
            self.episode_hist["total_reward"] += base_reward

            if self.game.is_player_dead():
                reward_val = -self.GOAL_REWARD * self.REWARD_SCALING
                total_reward += reward_val
                self.episode_hist["total_reward"] += reward_val
                self.episode_hist["goal_reward"] += reward_val
                
            elif timeout:
                reward_val = -self.GOAL_REWARD * self.REWARD_SCALING
                total_reward += reward_val
                self.episode_hist["total_reward"] += reward_val
                self.episode_hist["goal_reward"] += reward_val

            else:
                reward_val = self.GOAL_REWARD * self.REWARD_SCALING
                total_reward += reward_val
                self.episode_hist["total_reward"] += reward_val
                self.episode_hist["goal_reward"] += reward_val

            info["episode_stats"] = self.episode_hist

        return obs, total_reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.new_episode()
        self._reset_state_tracking()
        
        state = self.game.get_state()
        for _ in range(N_STACK):
            self.frame_buffer.append(state.screen_buffer)
            
        map_state = self.minimap.update(state.game_variables[2], state.game_variables[3])
        obs = flatten_observation(self.frame_buffer, state.game_variables, map_state)
        return obs, {}

    def close(self):
        self.game.close()

def load_expert_data_flat(path, env_num_vars):
    with open(path, "rb") as f:
        raw_data = pickle.load(f)

    obs_list, acts_list = [], []
    temp_map = DoomMiniMap()
    temp_buff = deque(maxlen=N_STACK)
    
    for obs, action, reward, next_obs in raw_data:
        game_vars = np.array(obs["gamevariables"], dtype=np.float32)
        screen = obs["screen"]
        
        if len(temp_buff) == 0:
            for _ in range(N_STACK): temp_buff.append(screen)
        else:
            temp_buff.append(screen)
            
        if game_vars.shape[0] > 3:
            map_state = temp_map.update(game_vars[2], game_vars[3])
        else:
            map_state = temp_map.state

        if game_vars.shape[0] < env_num_vars:
            pad = np.zeros(env_num_vars - game_vars.shape[0], dtype=np.float32)
            game_vars = np.concatenate([game_vars, pad])
        elif game_vars.shape[0] > env_num_vars:
            game_vars = game_vars[:env_num_vars]
            
        flat_obs = flatten_observation(temp_buff, game_vars, map_state)
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

def main():
    th.manual_seed(args.seed)
    np.random.seed(args.seed)
    expert_dir = "./doom_expert"
    model_save_path = "doom_agent_model"
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    print(f"Creating {N_ENVS} parallel environments...")
    venv = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)

    env_vars = venv.envs[0].num_vars

    model = PPO(
        "MlpPolicy",
        venv,
        policy_kwargs=dict(
            features_extractor_class=ThreeHeadExtractor,
            features_extractor_kwargs=dict(features_dim=512),
        ),
        verbose=1,
        learning_rate=linear_schedule(args.learning_rate),
        batch_size=args.ppo_batch_size,
        n_steps=args.ppo_n_steps,
        ent_coef=0.01,
    )

    policy = model.policy
    for param in policy.features_extractor.parameters():
        param.requires_grad = False
    
    if args.model != "NONE":
        temp_model = PPO.load(args.model)
        model.set_parameters(temp_model.get_parameters())

    log_csv_path = os.path.join(log_dir, "detailed_rewards.csv")
    reward_logger = DetailedRewardLogger(log_file=log_csv_path)

    if args.bc_train:
        print("--- Szkolenie na podstawie nagranej rozgrywki---")
        expert_set = os.listdir(expert_dir)
        policy = model.policy
        for param in policy.features_extractor.parameters():
            param.requires_grad = True
            
        for expert_gameplay in expert_set:
            try:
                transitions = load_expert_data_flat(f"./{expert_dir}/{expert_gameplay}", env_vars)
                rng = np.random.RandomState(args.seed)
                bc_trainer = BC(
                    observation_space=venv.observation_space,
                    action_space=venv.action_space,
                    policy=policy,
                    demonstrations=transitions,
                    rng=rng,
                    batch_size = len(transitions.obs) - len(transitions.obs)%args.bc_minibatch_size,
                    minibatch_size = args.bc_minibatch_size
                )
                bc_trainer.train(n_epochs=args.bc_epochs)
            except Exception as e:
                print(f"Skipping {expert_gameplay}: {e}")
        model.save(model_save_path+"BC_only")

    print("--- Szkolenie ---")

    if args.ppo_train:
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        eval_env = make_vec_env(make_env, n_envs=1)
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=log_dir,
            log_path=log_dir,
            eval_freq=args.eval_frequency,
            deterministic=True,
            render=False,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=args.eval_frequency,
            save_path=ckpt_dir,
            name_prefix="doom_model"
        )

        callbacks = [eval_callback, reward_logger, checkpoint_callback]

        model.learn(total_timesteps=args.ppo_timesteps, callback=callbacks)

    model.save(model_save_path)
    log_png = os.path.join(log_dir, "training_summary.png")
    plot_training_results(log_file=log_csv_path, output_img=log_png)
    venv.close()

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_train", action="store_true")
    parser.add_argument("--ppo_train", action="store_true")
    parser.add_argument("--bc_epochs", type=int, default=10)
    parser.add_argument("--ppo_timesteps", type=int, default=3000000)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--model", type=str, default="NONE")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--eval_frequency", type=int, default=7000)
    parser.add_argument("--bc_minibatch_size", type=int, default=128)
    parser.add_argument("--ppo_n_steps", type=int, default=4096)
    parser.add_argument("--ppo_batch_size", type=int, default=256)

    parser.add_argument("--goal_reward", type=float, default=5000.0, help="Reward for achieving the goal")
    parser.add_argument("--kill_reward", type=float, default=1000.0, help="Reward for killing an enemy or target")
    parser.add_argument("--hit_reward", type=float, default=200.0, help="Reward for hitting a target")
    parser.add_argument("--item_reward", type=float, default=20.0, help="Reward for collecting an item")
    parser.add_argument("--move_reward", type=float, default=60.0, help="Reward for movement")
    parser.add_argument("--secret_reward", type=float, default=2500.0, help="Reward for movement")

    parser.add_argument("--missed_shot_penalty", type=float, default=-200.0, help="Penalty for missed shot")
    parser.add_argument("--wall_stuck_penalty", type=float, default=-50.0, help="Penalty for getting stuck at a wall")
    parser.add_argument("--hit_taken_penalty", type=float, default=-500.0, help="Penalty for taking damage")
    parser.add_argument("--stay_on_visited_penalty", type=float, default=-15.0, help="Penalty for staying on visited cells")

    parser.add_argument("--min_distance_before_penalty", type=float, default=3.5, help="Minimum distance before penalty is applied")
    parser.add_argument("--map_cell_size", type=float, default=64.0, help="Size of the map cells")
    parser.add_argument("--reward_scaling", type=float, default=0.001, help="Scaling factor for rewards")

    args = parser.parse_args()
    N_ENVS = args.envs

    main()