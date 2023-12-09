from envs import BaseEnv
from maps import *
from minigrid.manual_control import ManualControl
from utils.shared_list import SharedList

class CustomManualControl(ManualControl):
    def reset(self, seed=None):
        self.env.reset(seed=seed, options=None)
        self.env.render()

env = BaseEnv(map_index=1, shared_list_actor=SharedList.remote(), render_mode="human")

# enable manual control for testing
manual_control = CustomManualControl(env, seed=42)
manual_control.start()
