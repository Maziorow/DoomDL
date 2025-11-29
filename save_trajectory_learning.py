import gymnasium
import cv2
import pickle
import keyboard  # New library for responsive input
from gymnasium.envs.registration import register

NUMBER_OF_EPISODES = 2
END_ON_ESC = False

# 1. Register: Use rgb_array to avoid VizDoom popping up its own secondary window
register(
    id="doom_e1m1",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": "env_configurations/doom_min.cfg"},
)

# 2. Initialize with rgb_array (faster, lets CV2 handle the only window)
env = gymnasium.make("doom_e1m1", render_mode="rgb_array")

trajectories = []

print("Controls: W, S, A, D, Space (Shoot), E, Q, U. ESC to finish.")

try:
    for episode in range(NUMBER_OF_EPISODES):
        obs, info = env.reset()
        done = False
        
        while not done:
            screen = obs["screen"]
            
            # Rendering
            cv2.imshow("Doom", screen)
            # We still need waitKey for cv2 to repaint the window, 
            # but we reduce it to 1ms and ignore the return value for input.
            cv2.waitKey(1) 

            # 3. Responsive Input Handling
            # This checks the state of the hardware key INSTANTLY
            action = 0
            if keyboard.is_pressed('w'): action = 4
            elif keyboard.is_pressed('s'): action = 3
            elif keyboard.is_pressed('a'): action = 8
            elif keyboard.is_pressed('d'): action = 7
            elif keyboard.is_pressed('space'): action = 6
            elif keyboard.is_pressed('e'): action = 1
            elif keyboard.is_pressed('q'): action = 2
            elif keyboard.is_pressed('u'): action = 5
            elif keyboard.is_pressed('esc'): 
                done = True

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Save trajectory
            trajectories.append((obs, action, reward, next_obs))
            obs = next_obs
            
            done = END_ON_ESC or not END_ON_ESC and (terminated or truncated)
            if done and keyboard.is_pressed('esc'):
                break 

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    env.close()
    cv2.destroyAllWindows()
    
    # Save results
    with open("doom_expert.pkl", "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved {len(trajectories)} steps.")