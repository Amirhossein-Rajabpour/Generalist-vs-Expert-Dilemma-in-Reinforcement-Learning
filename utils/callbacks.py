import csv
from typing import Dict

import ray
from ray.tune.utils import flatten_dict
from ray.tune.experiment.trial import Trial
from ray.tune.logger import CSVLoggerCallback

from envs import all_maps


class CustomLoggerCallback(CSVLoggerCallback):
    def __init__(self, shared_list_actor):
        super().__init__()
        self.shared_list_actor = shared_list_actor
        
    def log_trial_result(self, iteration: int, trial: "Trial", result: Dict):
        rewards = ray.get(self.shared_list_actor.rewards_get_list.remote())
        total_rewards = ray.get(self.shared_list_actor.total_rewards_get_list.remote())
        eposides = ray.get(self.shared_list_actor.eposides_get_list.remote())
        total_eposides = ray.get(self.shared_list_actor.total_eposides_get_list.remote())
        time_steps = ray.get(self.shared_list_actor.time_steps_get_list.remote())
        total_time_steps = ray.get(self.shared_list_actor.total_time_steps_get_list.remote())
        
        new_cols = []
        for i in range(len(all_maps)):
            if eposides[i] != 0:
                col_mean_reward = f'env{i}_mean_reward'
                col_total_mean_reward = f'env{i}_total_mean_reward'
                col_mean_length = f'env{i}_mean_episode_length'
                col_total_time_steps = f'env{i}_tts'
                result[col_mean_reward] = rewards[i]/eposides[i]
                result[col_total_mean_reward] = total_rewards[i]/total_eposides[i]
                result[col_mean_length] = time_steps[i]/eposides[i]
                result[col_total_time_steps] = total_time_steps[i]
                new_cols.append(col_mean_reward)
                new_cols.append(col_total_mean_reward)
                new_cols.append(col_mean_length)
                new_cols.append(col_total_time_steps)
            
        if trial not in self._trial_files:
            self._setup_trial(trial)

        tmp = result.copy()
        tmp.pop("config", None)
        result = flatten_dict(tmp, delimiter="/")

        if not self._trial_csv[trial]:
            self._trial_csv[trial] = csv.DictWriter(
                self._trial_files[trial], result.keys()
            )
            if not self._trial_continue[trial]:
                self._trial_csv[trial].writeheader()
        
        fieldnames = list(self._trial_csv[trial].fieldnames) + new_cols
        self._trial_csv[trial].writerow(
            {k: v for k, v in result.items() if k in fieldnames}
        )
        self._trial_files[trial].flush()
        self.shared_list_actor.reset.remote()