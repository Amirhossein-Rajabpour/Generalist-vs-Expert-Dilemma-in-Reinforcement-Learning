from __future__ import annotations
from typing import Any

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv

import maps

all_maps = [getattr(maps, m) for m in dir(maps) if not m.startswith("__")]

class BaseEnv(MiniGridEnv):
    def __init__(
        self,
        map_index,
        shared_list_actor = None,
        size=10,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps=200,
        **kwargs,
    ):
        self.map_index = map_index
        self.map = all_maps[map_index]
        self.shared_list_actor = shared_list_actor
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate verical separation wall
        for i in range(width - 2):
            for j in range(height - 2):
                if self.map[i][j] == 1:
                    self.grid.set(j+1, i+1, Wall())
        
        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"
        
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.shared_list_actor.rewards_add_item.remote(self.map_index, reward)
        self.shared_list_actor.time_steps_increment.remote(self.map_index)

        return obs, reward, terminated, truncated, info
    
    def reset(self, *, seed, options):
        self.shared_list_actor.eposides_increment.remote(self.map_index)
        return super().reset(seed=seed, options=options)
