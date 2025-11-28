import pickle
import numpy as np
import gymnasium
from gymnasium import ObservationWrapper, spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
import cv2

# Wczytanie środowiska
register(
    id="doom_e1m1",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": "env_configurations/doom_min.cfg"},
)


# Pokazanie gry
def show_obs(obs, gv_size=2, screen_shape=(240, 320, 3)):
    flat_screen = obs[gv_size:].reshape((3, screen_shape[0], screen_shape[1]))
    screen_hwc = flat_screen.transpose(1, 2, 0).astype(np.uint8)
    cv2.imshow("Doom screen", screen_hwc)
    cv2.waitKey(1)


# Wrapper na gamevariables i screen (nie akceptuje dicta BC)
class FlattenObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        gv_space = env.observation_space["gamevariables"]
        sc_space = env.observation_space["screen"]
        chw_shape = (sc_space.shape[2], sc_space.shape[0], sc_space.shape[1])
        self.screen_size = int(np.prod(chw_shape))
        self.gv_size = int(np.prod(gv_space.shape))
        low = np.concatenate([
            np.full(self.gv_size, gv_space.low.min(), dtype=np.float32),
            np.full(self.screen_size, sc_space.low.min(), dtype=np.float32),
        ])
        high = np.concatenate([
            np.full(self.gv_size, gv_space.high.max(), dtype=np.float32),
            np.full(self.screen_size, sc_space.high.max(), dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    # Wrzucenie do listy gamevariables i screen po spłaszczeniu wymiarów
    def observation(self, obs):
        gv = np.array(obs["gamevariables"], dtype=np.float32).reshape(-1)
        sc = np.transpose(obs["screen"], (2, 0, 1)).astype(np.float32).reshape(-1)
        return np.concatenate([gv, sc], dtype=np.float32)


def load_demonstrations(pkl_path="doom_expert.pkl"):
    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)
    obs_list = []
    next_obs_list = []
    acts_list = []
    # Spłaszczenie wymiarów
    for obs, action, reward, next_obs in trajectories:
        gv = np.array(obs["gamevariables"], dtype=np.float32).reshape(-1)
        sc = np.transpose(obs["screen"], (2, 0, 1)).astype(np.float32).reshape(-1)
        flat_obs = np.concatenate([gv, sc], dtype=np.float32)
        ngv = np.array(next_obs["gamevariables"], dtype=np.float32).reshape(-1)
        nsc = np.transpose(next_obs["screen"], (2, 0, 1)).astype(np.float32).reshape(-1)
        flat_next_obs = np.concatenate([ngv, nsc], dtype=np.float32)
        obs_list.append(flat_obs)
        next_obs_list.append(flat_next_obs)
        acts_list.append(action)

    # Przygotowanie danych dla Transitions
    obs_array = np.stack(obs_list)
    next_obs_array = np.stack(next_obs_list)
    acts_array = np.array(acts_list, dtype=np.int64)
    dones = np.zeros(len(acts_array), dtype=bool)
    infos = [{} for _ in range(len(acts_array))]
    transitions = Transitions(
        obs=obs_array,
        acts=acts_array,
        next_obs=next_obs_array,
        dones=dones,
        infos=infos,
    )
    return transitions


def train_bc_model(
    save_path="doom_bc_model", demos_path="doom_expert.pkl", bc_epochs=1
):
    env = FlattenObsWrapper(gymnasium.make("doom_e1m1", render_mode=None))
    # Dla spłaszczonych danych - MlpPolicy; MultiInputPolicy - zwraca błędy potem w BC
    # bo BC nie chce dicta tylko typ float etc.
    base_model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4)
    demos = load_demonstrations(demos_path)
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=base_model.policy,
        demonstrations=demos,
        rng=np.random.default_rng(),
    )
    bc_trainer.train(n_epochs=bc_epochs)
    base_model.save(save_path)
    env.close()
    print(f"Model saved to: {save_path}")


def evaluate_model(model_path="doom_bc_model", episodes=3):
    env = FlattenObsWrapper(gymnasium.make("doom_e1m1", render_mode="human"))
    model = PPO.load(model_path)
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            # show_obs(obs)
            total_reward += reward
            steps += 1
        print(f"Episode {ep + 1}: steps={steps}, reward={total_reward}")
    env.close()


if __name__ == "__main__":
    train_bc_model(
        save_path="doom_bc_model", demos_path="doom_expert.pkl", bc_epochs=10
    )
    evaluate_model(model_path="doom_bc_model", episodes=3)
