import gymnasium
import cv2
import pickle
from gymnasium.envs.registration import register

# Liczba epizodów do przegrania
NUMBER_OF_EPISODES = 9

# Wybór czy ma user kończyć czy środowisko (True = user)
END_ON_ESC = True

# Wczytanie środowiska
register(
    id="doom_e1m1",
    entry_point="vizdoom.gymnasium_wrapper.base_gymnasium_env:VizdoomEnv",
    kwargs={"level": "env_configurations/doom_min.cfg"},
)

# Utworzenie środowiska z okienkiem gry
env = gymnasium.make("doom_e1m1", render_mode="human")

# Trajektorie akcji podejmowanych przez model
trajectories = []

for episode in range(NUMBER_OF_EPISODES):
    # rozpoczęcie gry od nowa
    obs, info = env.reset()
    done = False
    while not done:
        screen = obs["screen"]
        cv2.imshow("Doom", screen)
        key = cv2.waitKey(1)
        # Key binding (kolejność jest od dołu do góry z .cfg)
        if key == ord("w"):
            action = 4
        elif key == ord("s"):
            action = 3
        elif key == ord("a"):
            action = 8
        elif key == ord("d"):
            action = 7
        elif key == ord(" "):
            action = 6
        elif key == ord("e"):
            action = 1
        elif key == ord("q"):
            action = 2
        elif key == ord("u"):
            action = 5
        elif key == 27: # esc
            done = True
        else:
            action = 0
        next_obs, reward, terminated, truncated, info = env.step(action)
        trajectories.append((obs, action, reward, next_obs))
        obs = next_obs
        done = END_ON_ESC or not END_ON_ESC and (terminated or truncated)

env.close()
cv2.destroyAllWindows()

# Zapis do pkl rozgrywek
with open("doom_expert.pkl", "wb") as f:
    pickle.dump(trajectories, f)
