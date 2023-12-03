from matplotlib import pyplot as plt
from envs import BaseEnv
from maps import *
from minigrid.manual_control import ManualControl


env = BaseEnv(map=map3, render_mode="human")

# enable manual control for testing
manual_control = ManualControl(env, seed=42)
manual_control.start()
