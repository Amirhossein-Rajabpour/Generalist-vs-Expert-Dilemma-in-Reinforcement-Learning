import ray
from envs import all_maps

@ray.remote(num_cpus=5)
class SharedList:
    def __init__(self):
        self.num_envs = len(all_maps)
        self.rewards = [0] * self.num_envs
        self.total_rewards = [0] * self.num_envs
        self.eposides = [0] * self.num_envs
        self.total_eposides = [0] * self.num_envs
        self.time_steps = [0] * self.num_envs
        self.total_time_steps = [0] * self.num_envs

    def rewards_add_item(self, index, item):
        self.rewards[index] += item
        self.total_rewards[index] += item

    def rewards_get_list(self):
        return self.rewards
    
    def total_rewards_get_list(self):
        return self.total_rewards
    
    def eposides_increment(self, index):
        self.eposides[index] += 1
        self.total_eposides[index] += 1
        
    def eposides_get_list(self):
        return self.eposides
    
    def total_eposides_get_list(self):
        return self.total_eposides
    
    def time_steps_increment(self, index):
        self.time_steps[index] += 1
        self.total_time_steps[index] += 1
        
    def time_steps_get_list(self):
        return self.time_steps
    
    def total_time_steps_get_list(self):
        return self.total_time_steps
    
    def reset(self):
        for i in range(self.num_envs):
            self.rewards[i] = 0
            self.eposides[i] = 0
            self.time_steps[i] = 0