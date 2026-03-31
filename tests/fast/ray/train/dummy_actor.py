"""Lightweight Ray actor for unit testing RayTrainCell/RayTrainGroup without GPU or real training."""

import ray


@ray.remote(num_gpus=0, num_cpus=0)
class DummyTrainActor:

    def init(self, **kwargs):
        pass

    def reconfigure_indep_dp(self, **kwargs):
        pass

    def send_ckpt(self, **kwargs):
        pass

    def train(self, **kwargs):
        pass

    def set_rollout_manager(self, manager):
        pass

    def wake_up(self):
        pass

    def sleep(self):
        pass

    def clear_memory(self):
        pass

    def save_model(self, *args, **kwargs):
        pass

    def update_weights(self):
        pass
