from matplotlib import pyplot as plt
from envs import BaseEnv
from maps import *
from minigrid.manual_control import ManualControl


env = BaseEnv(map=map3, render_mode="human")

# plot initial state
observation, info = env.reset()
img = env.render()
plt.imshow(img)
plt.savefig("initial_state.png")

# enable manual control for testing
manual_control = ManualControl(env, seed=42)
manual_control.start()
