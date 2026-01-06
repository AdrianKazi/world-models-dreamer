import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape):
        self.capacity = capacity

        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=bool)

        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity) 

    def sample(self, batch_size, seq_len):
        idxs = np.random.randint(0, self.size - seq_len, size=batch_size)
        obs = np.stack([self.obs[i:i+seq_len] for i in idxs])
        actions = np.stack([self.actions[i:i+seq_len] for i in idxs])
        rewards = np.stack([self.rewards[i:i+seq_len] for i in idxs])
        dones = np.stack([self.dones[i:i+seq_len] for i in idxs])
        return obs, actions, rewards, dones