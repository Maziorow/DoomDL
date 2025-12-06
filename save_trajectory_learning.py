import os
import vizdoom as vzd
import cv2
import pickle
import keyboard
import numpy as np

CONFIG_PATH = "env_configurations/doom_min.cfg"
EPISODES_TO_RECORD = 3
DIR_NAME = "doom_expert"
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

        px = game_vars[2]
        py = game_vars[3]
        
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

def main():
    game = vzd.DoomGame()
    game.load_config(CONFIG_PATH)
    game.set_window_visible(False) 
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.init()

    state = game.get_state()
    vars_count = len(state.game_variables)
    print(f"Game Variables Found: {vars_count}")
    
    if vars_count < 4:
        print("WARNING: It looks like POSITION_X and POSITION_Y are missing!")
        print("Please check available_game_variables in doom_min.cfg")
        input("Press Enter to continue anyway (or Ctrl+C to fix config)...")

    action_count = len(game.get_available_buttons())
    print(f"Actions initialized: {action_count}")
    print("Controls: W, S, A, D, Space (Shoot), E (Turn R), U (Use), Q (Turn L). ESC to Quit.")

    minimap = MinimapViz()

    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    
    try:
        for episode in range(EPISODES_TO_RECORD):
            trajectories = []
            output_file = f"{DIR_NAME}/doom_expert{episode}.pkl"
            print(f"--- Recording Episode {episode + 1} ---")
            game.new_episode()
            
            minimap.path = [] 
            minimap.canvas[:] = 0 

            while not game.is_episode_finished():
                state = game.get_state()
                screen = state.screen_buffer
                game_vars = state.game_variables

                cv2.imshow("Doom Recorder", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
                map_img = minimap.update(game_vars)
                cv2.imshow("Minimap", map_img)
                cv2.waitKey(1)

                actions = [False] * action_count
                
                if keyboard.is_pressed('w'): actions[4] = True
                if keyboard.is_pressed('s'): actions[5] = True
                if keyboard.is_pressed('a'): actions[0] = True
                if keyboard.is_pressed('d'): actions[1] = True
                
                if keyboard.is_pressed('q'): actions[6] = True
                if keyboard.is_pressed('e'): actions[7] = True
                
                if keyboard.is_pressed('space'): actions[2] = True
                if keyboard.is_pressed('u'): actions[3] = True

                if keyboard.is_pressed('esc'):
                    print("Recording cancelled.")
                    return

                reward = game.make_action(actions)

                action_index = 8
                
                if actions[2]: action_index = 2
                elif actions[3]: action_index = 3
                elif actions[4]: action_index = 4
                elif actions[5]: action_index = 5
                elif actions[0]: action_index = 0
                elif actions[1]: action_index = 1
                elif actions[6]: action_index = 6
                elif actions[7]: action_index = 7

                if action_index != 8:
                    next_state_data = None
                    if not game.is_episode_finished():
                        ns = game.get_state()
                        next_state_data = {
                            "screen": ns.screen_buffer, 
                            "gamevariables": ns.game_variables
                        }

                    current_obs = {
                        "screen": screen, 
                        "gamevariables": game_vars
                    }
                    
                    trajectories.append((current_obs, action_index, reward, next_state_data))

            print(f"Episode {episode + 1} Complete.")
            clean_trajectories = [t for t in trajectories if t[3] is not None]

            with open(output_file, "wb") as f:
                pickle.dump(clean_trajectories, f)
            print(f"Saved {len(clean_trajectories)} steps to {output_file}.")

    finally:
        game.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
