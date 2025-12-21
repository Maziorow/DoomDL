import os
import vizdoom as vzd
import cv2
import pickle
import keyboard
import numpy as np
import math

CONFIG_PATH = "env_configurations/doom_min.cfg"
EPISODES_TO_RECORD = 5
DIR_NAME = "doom_expert"

class MinimapViz:
    def __init__(self, width=500, height=500, scale=1.0):
        self.w = width
        self.h = height
        self.scale = scale
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.path = []
        self.WORLD_MAX = 8192
        self.WORLD_MIN = -8192

    def update(self, game_vars):
        if len(game_vars) < 4: return self.canvas
        px = game_vars[2]
        py = game_vars[3]
        effective_scale = 15
        sx = int((self.w // 2) + (px / effective_scale))
        sy = int((self.h // 2) - (py / effective_scale))

        if 0 <= sx < self.w and 0 <= sy < self.h:
            self.path.append((sx, sy))

        self.canvas[:] = 0
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                cv2.line(self.canvas, self.path[i-1], self.path[i], (255, 255, 255), 1)

        display_img = self.canvas.copy()
        cv2.circle(display_img, (sx, sy), 3, (0, 0, 255), -1)

        if len(self.path) > 0:
            cv2.circle(display_img, self.path[0], 2, (0, 255, 0), -1)

        coord_text = f"World X: {px:.1f}, World Y: {py:.1f}"
        cv2.putText(display_img, coord_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return display_img

class RewardCalculator:
    def __init__(self):
        self.GOAL_REWARD = 5000.0
        self.SECRET_REWARD = 2500.0
        self.KILL_REWARD = 1000.0
        self.HIT_REWARD = 200.0
        self.ITEM_REWARD = 20.0
        self.MOVE_REWARD = 200.0
        self.MISSED_SHOT_PENALTY = -100.0
        self.WALL_STUCK_PENALTY = -200.0
        self.MIN_DISTANCE_BEFORE_PENALTY = 20.0
        self.HIT_TAKEN_PENALTY = -500.0
        self.MAP_CELL_SIZE = 64.0
        self.REWARD_SCALING = 0.001
        self.MAX_ACTION_NUMBER = 7000 
        self.MAX_BUFF_SIZE = 10

        self.visited_cells = {}
        self.last_cell = None
        self.stuck_timer = 0
        self.c_x_buff = []
        self.c_y_buff = []

        self.last_health = 100
        self.last_ammo = 50
        self.last_hit_count = 0
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

    def reset(self, initial_vars):
        self.visited_cells = {}
        self.last_cell = None
        self.stuck_timer = 0
        self.c_x_buff = []
        self.c_y_buff = []
        
        if initial_vars is not None:
            self.last_health = initial_vars[0]
            self.last_ammo = initial_vars[1]
            self.last_hit_count = initial_vars[4] if len(initial_vars) > 4 else 0
            self.last_kill_count = initial_vars[7] if len(initial_vars) > 7 else 0
            self.last_secret_count = initial_vars[6] if len(initial_vars) > 6 else 0
        else:
            self.last_health = 100
            self.last_ammo = 50
            self.last_hit_count = 0
            self.last_kill_count = 0
            self.last_secret_count = 0

    def calculate_step_reward(self, current_vars, base_reward, done, episode_time):
        if done:
            total_reward = base_reward
            timeout = episode_time >= (self.MAX_ACTION_NUMBER - 1)
            if self.last_health > 0 and not timeout:
                total_reward += self.GOAL_REWARD * self.REWARD_SCALING
            elif self.last_health > 0 and timeout:
                total_reward += -self.GOAL_REWARD * self.REWARD_SCALING
            return total_reward

        reward = 0.0
        c_health = current_vars[0]
        c_ammo = current_vars[1]
        c_x, c_y = current_vars[2], current_vars[3]
        c_hits = current_vars[4] if len(current_vars) > 4 else 0
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
            self.stuck_timer = 10
            print("Stuck penalty!")

        

        ammo_used = self.last_ammo - c_ammo
        damage_dealt = c_hits - self.last_hit_count

        if ammo_used > 0:
            if damage_dealt <= 0:
                reward += self.MISSED_SHOT_PENALTY * self.REWARD_SCALING

        if c_kills > self.last_kill_count:
            r = (c_kills - self.last_kill_count) * self.KILL_REWARD
            reward += r * self.REWARD_SCALING

        if c_secrets > self.last_secret_count:
            reward += (self.SECRET_REWARD * self.REWARD_SCALING) * (c_secrets - self.last_secret_count)

        if damage_dealt > 0:
            r = damage_dealt * self.HIT_REWARD
            reward += r * self.REWARD_SCALING

        if c_health < self.last_health:
            reward += self.HIT_TAKEN_PENALTY * self.REWARD_SCALING
            print("Hit taken penalty!")

        cell_x = int(c_x / self.MAP_CELL_SIZE)
        cell_y = int(c_y / self.MAP_CELL_SIZE)
        current_cell = (cell_x, cell_y)
        if current_cell != self.last_cell:
            if current_cell not in self.visited_cells:
                self.visited_cells[current_cell] = True
                reward += self.MOVE_REWARD * self.REWARD_SCALING
                print(f"Visited cell {current_cell} for the first time!")
            self.last_cell = current_cell

        self.last_health = c_health
        self.last_ammo = c_ammo
        self.last_hit_count = c_hits
        self.last_kill_count = c_kills
        self.last_secret_count = c_secrets

        return (base_reward * self.REWARD_SCALING) + reward

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
    reward_calc = RewardCalculator()

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
            
            initial_state = game.get_state()
            reward_calc.reset(initial_state.game_variables)

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

                base_reward = game.make_action(actions)
                done = game.is_episode_finished()
                episode_time = game.get_episode_time()

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
                    next_vars = None
                    
                    if not done:
                        ns = game.get_state()
                        next_vars = ns.game_variables
                        next_state_data = {
                            "screen": ns.screen_buffer, 
                            "gamevariables": ns.game_variables
                        }
                    else:
                        next_vars = game_vars 

                    calculated_reward = reward_calc.calculate_step_reward(next_vars, base_reward, done, episode_time)

                    current_obs = {
                        "screen": screen, 
                        "gamevariables": game_vars
                    }
                    
                    trajectories.append((current_obs, action_index, calculated_reward, next_state_data))

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