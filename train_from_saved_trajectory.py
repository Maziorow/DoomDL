import pickle
import numpy as np
import gymnasium
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions

register(
    id="doom_e1m1",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": "env_configurations/doom_min.cfg"},
)


def load_demonstrations(pkl_path="doom_expert.pkl"):
    with open(pkl_path, "rb") as f:
        trajectories = pickle.load(f)

    gamevars = []
    screens = []
    acts_list = []
    next_gamevars = []
    next_screens = []

    for obs, action, reward, next_obs in trajectories:
        gamevars.append(np.array(obs["gamevariables"], dtype=np.float32))
        screens.append(np.transpose(obs["screen"], (2, 0, 1)))
        next_gamevars.append(np.array(next_obs["gamevariables"], dtype=np.float32))
        next_screens.append(np.transpose(next_obs["screen"], (2, 0, 1)))
        acts_list.append(action)
        
    obs_array = np.array([
        {"gamevariables": gv, "screen": sc} for gv, sc in zip(gamevars, screens)
    ])
    next_obs_array = np.array([
        {"gamevariables": gv, "screen": sc}
        for gv, sc in zip(next_gamevars, next_screens)
    ])
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
    save_path="doom_bc_model", demos_path="doom_expert.pkl", bc_epochs=10
):
    env = gymnasium.make("doom_e1m1", render_mode=None)
    base_model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=1e-4)
    print("Env observation_space:", env.observation_space)
    print("Policy observation_space:", base_model.policy.observation_space)

    demos = load_demonstrations(demos_path)
    bc_trainer = BC(
        observation_space=base_model.policy.observation_space,
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
    env = gymnasium.make("doom_e1m1", render_mode="human")
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
            total_reward += reward
            steps += 1
        print(f"Episode {ep + 1}: steps={steps}, reward={total_reward}")
    env.close()


if __name__ == "__main__":
    train_bc_model(
        save_path="doom_bc_model", demos_path="doom_expert.pkl", bc_epochs=10
    )
    evaluate_model(model_path="doom_bc_model", episodes=3)
