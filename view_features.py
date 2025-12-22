import keyboard
import vizdoom as vzd
import cv2
import numpy as np
import torch as th
import torch.nn as nn

SCREEN_W, SCREEN_H = 60, 45
SCREEN_CHANNELS = 3
CONFIG_PATH = "env_configurations/doom_min.cfg"


class DoomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def preprocess_screen(screen):
    resized = cv2.resize(screen, (SCREEN_W, SCREEN_H), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    chw = np.transpose(norm, (2, 0, 1))
    tensor = th.from_numpy(chw).unsqueeze(0)  
    return tensor


def show_feature_maps(feature_tensor, win_name="CNN Features"):
    features = feature_tensor.squeeze(0).detach().cpu().numpy() 
    num_maps = features.shape[0]

    grid_cols = 8
    grid_rows = num_maps // grid_cols

    maps = []
    for i in range(num_maps):
        fmap = features[i]
        fmap -= fmap.min()
        fmap /= fmap.max() + 1e-6
        fmap = np.uint8(fmap * 255)
        fmap = cv2.resize(fmap, (80, 80))
        maps.append(fmap)

    rows = []
    for r in range(grid_rows):
        row = maps[r * grid_cols : (r + 1) * grid_cols]
        rows.append(np.hstack(row))

    grid = np.vstack(rows)
    cv2.imshow(win_name, grid)


def main():
    game = vzd.DoomGame()
    game.load_config(CONFIG_PATH)
    game.set_window_visible(True)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.init()

    cnn = DoomCNN().eval()

    print("ESC – wyjście")

    button_count = len(game.get_available_buttons())

    while not game.is_episode_finished():
        state = game.get_state()
        screen = state.screen_buffer

        # === PREPROCESS + CNN ===
        input_tensor = preprocess_screen(screen)
        with th.no_grad():
            features = cnn(input_tensor)

        # === WIZUALIZACJA ===
        cv2.imshow("Doom Screen", cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
        show_feature_maps(features)

        # === STEROWANIE ===
        actions = [False] * button_count

        if keyboard.is_pressed("w"):
            actions[4] = True  # MOVE_FORWARD
        if keyboard.is_pressed("s"):
            actions[5] = True  # MOVE_BACKWARD
        if keyboard.is_pressed("a"):
            actions[0] = True  # MOVE_LEFT
        if keyboard.is_pressed("d"):
            actions[1] = True  # MOVE_RIGHT

        if keyboard.is_pressed("q"):
            actions[6] = True  # TURN_LEFT
        if keyboard.is_pressed("e"):
            actions[7] = True  # TURN_RIGHT

        if keyboard.is_pressed("space"):
            actions[2] = True  # ATTACK
        if keyboard.is_pressed("u"):
            actions[3] = True  # USE

        if keyboard.is_pressed("esc"):
            break

        game.make_action(actions)

        cv2.waitKey(1)

    game.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
