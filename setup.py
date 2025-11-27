import gymnasium
import os
from vizdoom import gymnasium_wrapper
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import cv2

register(
    id="doom_e1m1",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": "env_configurations/doom_min.cfg"},
)

def train_model(save_path="doom_ppo_model"):
    print("--- Starting Training (This may take a while) ---")
    env = gymnasium.make("doom_e1m1", render_mode=None)
    
    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=0.0001)
    # There is need for MUCH more episodes
    model.learn(total_timesteps=5)
    
    model.save(save_path)
    env.close()

class DoomEnv():
    def __init__(self, model_path, scenario_name, render_mode="human"):
        self.env = gymnasium.make(scenario_name, render_mode=render_mode)
        
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)

    def policy(self, observation):
        action, _ = self.model.predict(observation, deterministic=True)
        return action

    def start_run(self, episodes):
        for episode_num in range(episodes):
            observation, info = self.env.reset()
            
            terminated = False
            truncated = False
            total_reward = 0
            steps = 0

            while not (terminated or truncated):
                action = self.policy(observation)
                screen = observation["screen"]
                screen_grayscale = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
                cv2.imshow("Doom in grayscale (Ready to process video stream)", screen_grayscale)
                observation, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                steps += 1

            print(f"Episode {episode_num + 1} finished after {steps} steps. Score: {total_reward}")

        self.env.close()

def main():
    train_model("doom_ppo_model")

    DoomSetup = DoomEnv("doom_ppo_model", "doom_e1m1", render_mode="human")
    DoomSetup.start_run(5)

if __name__ == "__main__":
    main()